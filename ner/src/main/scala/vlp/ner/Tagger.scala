package vlp.ner

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths, StandardOpenOption}

import vlp.VLP
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.slf4j
import org.slf4j.LoggerFactory
import scopt.OptionParser

import scala.collection.mutable.ListBuffer
import scala.io.Source


/**
 * @author Phuong LE-HONG
 * <p>
 * Sep 5, 2016, 12:22:53 PM
 * <p>
 * Implementation of a sequence tagger.
 */
class Tagger(sparkSession: SparkSession, config: ConfigNER) {
  val logger: slf4j.Logger = LoggerFactory.getLogger(getClass.getName)

  lazy val forwardModel = PipelineModel.load(config.modelPath + config.language + "/cmm-f")
  lazy val backwardModel = PipelineModel.load(config.modelPath + config.language + "/cmm-b")

  lazy val partOfSpeechTagger = new vlp.tag.Tagger(sparkSession, vlp.tag.ConfigPoS())
  lazy val partOfSpeechModel = PipelineModel.load(vlp.tag.ConfigPoS().modelPath)

  private def createDF(sentences: List[Sentence]): Dataset[LabeledContext] = {
    val contexts = sentences.flatMap {
      sentence => Featurizer.extract(sentence)
    }
    import sparkSession.implicits._
    sparkSession.createDataFrame(contexts).as[LabeledContext]
  }

  /**
    * Trains a MLR model: (sentences, model parameters) => model.
    *
    * @param sentences list of training sentences
    * @return a pipeline model.
    */
  def train(sentences: List[Sentence]): PipelineModel = {
    VLP.log("Preparing data frame for training... Please wait.")
    val trainingSentences = if (!config.reversed) {
      sentences
    } else {
      sentences.map(s => Sentence(s.tokens.reverse))
    }
    val df = createDF(trainingSentences)
    df.cache()

    if (config.verbose) {
      VLP.log("#(sentences) = " + sentences.size)
      VLP.log("#(contexts) = " + df.count())
      VLP.log("#(numFeatures) = " + config.numFeatures)
      df.show(10, truncate = false)
    }

    // create and fit a processing pipeline 
    val labelIndexer = new StringIndexer().setInputCol("tag").setOutputCol("label")
    val tokenizer = new Tokenizer().setInputCol("bof").setOutputCol("tokens")
    val hashingTF = new HashingTF().setInputCol("tokens").setOutputCol("features").setBinary(true).setNumFeatures(config.numFeatures)
    val mlr = new LogisticRegression().setMaxIter(config.iterations).setRegParam(config.lambda).setStandardization(false).setTol(1E-5)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, hashingTF, mlr))
    val model = pipeline.fit(df)

    // overwrite the trained pipeline
    val modelPath = config.modelPath + config.language + (if (config.reversed) "/cmm-b"; else "/cmm-f")
    model.write.overwrite().save(modelPath)
    // print some strings to debug the model
    if (config.verbose) {
      val labels = model.stages(0).asInstanceOf[StringIndexerModel].labels
      VLP.log("#(labels) = " + labels.length)
      val logreg = model.stages(3).asInstanceOf[LogisticRegressionModel]
      VLP.log(logreg.explainParams())
    }
    model
  }

  /**
    * Tags a list of sentences: (model, input sentences) => output sentences.
    *
    * @param sentences a list of sentences
    * @return a list of tagged sentences.
    */
  def tag(sentences: List[Sentence]): List[Sentence] = {
    val modelPath = config.modelPath + config.language + (if (config.reversed) "/cmm-b"; else "/cmm-f")
    val model = PipelineModel.load(modelPath)
    tag(model, sentences, config.reversed)
  }

  def tag(model: PipelineModel, sentences: List[Sentence], isReversed: Boolean): List[Sentence] = {
    val ss = if (!isReversed) sentences else {
      sentences.map(s => Sentence(s.tokens.reverse))
    }
    val ys = tag(model, ss)
    if (!isReversed) ys; else ys.map(s => Sentence(s.tokens.reverse))
  }

  def tag(model: PipelineModel, sentences: List[Sentence]): List[Sentence] = {
    val decoder = new Decoder(sparkSession, DecoderType.Greedy, model)
    decoder.decode(sentences)
  }

  def tag(decoder: Decoder, sentences: List[Sentence]): List[Sentence] = {
    decoder.decode(sentences)
  }

  /**
    * Tags a list of sentences and saves the result to an output file in a two-column format, which
    * is suitable for the evaluation tool 'conlleval' of the CoNLL-2003 NER shared-task.
    *
    * @param sentences a list of sentences to tag
    * @param outputPath an output path in conlleval format for evaluation
    */
  def tag(sentences: List[Sentence], outputPath: String) {
    val lines = new ListBuffer[String]()
    // copy sentences to xs to preserve gold NER tags
    val xs = sentences.map(x => Sentence(x.tokens.clone()))
    // tag sentences; their NER annotation will be updated
    val ys = tag(sentences)
    // prepare results of the format (gold tag, predicted tag)
    for (i <- ys.indices) {
      val x = xs(i)
      val y = ys(i)
      val line = ListBuffer[String]()
      for (j <- 0 until x.length) {
        line.append(x.tokens(j).annotation(Label.NamedEntity) + ' ' + y.tokens(j).annotation(Label.NamedEntity))
      }
      if (config.reversed)
        lines.append(line.reverse.mkString("\n"))
      else lines.append(line.mkString("\n"))
      lines.append("\n\n")
    }
    // save the lines
    val pw = new java.io.PrintWriter(new java.io.File(outputPath))
    try {
      lines.foreach(line => pw.write(line))
    } finally {
      pw.close()
    }
  }

  /**
    * Tests a machine learning model for NER on test sets.
    * @param sentences a list of sentences to test
    */
  def test(sentences: List[Sentence]): Unit = {
    VLP.log("#(sentences) = " + sentences.length)
    val outputPath = config.dataPath + ".out" + (if (config.reversed) ".b"; else ".f")
    tag(sentences, outputPath)
  }

  /**
    * Bi-directional processing in batch mode, where two models are load and results
    * are saved into an external file for conlleval script.
    * @param sentences a list of sentences to perform NER
    * @param outputPath a file in conlleval format (two columns).
    * @return a list of [[Sentence]]
    */
  def combine(sentences: List[Sentence], outputPath: String = ""): List[Sentence] = {
    val forwardDecoder = new Decoder(sparkSession, DecoderType.Greedy, forwardModel)
    val backwardDecoder = new Decoder(sparkSession, DecoderType.Greedy, backwardModel)
    val result = combine(sentences, forwardDecoder, backwardDecoder)
    // save the lines to a file if the output path is not empty
    if (outputPath.nonEmpty) {
      val lines = ListBuffer[String]()
      for (i <- sentences.indices) {
        val pair = (sentences(i).tokens) zip (result(i).tokens)
        val line = pair.map(p => p._1.namedEntity + ' ' + p._2.namedEntity).mkString("\n")
        lines.append(line)
        lines.append("\n\n")
      }
      val pw = new java.io.PrintWriter(new java.io.File(outputPath))
      try {
        lines.foreach(line => pw.write(line))
      } finally {
        pw.close()
      }
    }
    result
  }

  /**
    * Tags an input file by combining both forward and backward models.
    *
    * @param inputPath an input file in CoNLL-2003 format (VLSP format)
    * @param outputPath an output path for conlleval script
    */
  def combine(inputPath: String, outputPath: String): List[Sentence] = {
    val testSet = CorpusReader.readCoNLL(inputPath, config.twoColumns)
    VLP.log("#(sentences) = " + testSet.length)
    combine(testSet, outputPath)
  }

  /**
    * Finds named entities of a text using bidirectional inference.
    * @param sentences
    * @param forwardDecoder
    * @param backwardDecoder
    * @return a list of sentences
    */
  def combine(sentences: List[Sentence], forwardDecoder: Decoder, backwardDecoder: Decoder): List[Sentence] = {
    // forward tagging
    val us = sentences.map(x => Sentence(x.tokens.clone()))
    val ys = tag(forwardDecoder, us)
    // backward tagging
    val vs = sentences.map(x => Sentence(x.tokens.clone().reverse))
    val zs = tag(backwardDecoder, vs)
    // combine the result of ys and zs
    val result = ListBuffer[Sentence]()
    for (i <- ys.indices) {
      val y = ys(i)
      val z = Sentence(zs(i).tokens.reverse)
      val s = ListBuffer[Token]()
      for (j <- 0 until y.length)
        if (z.tokens(j).namedEntity.endsWith("LOC") && !y.tokens(j).namedEntity.endsWith("ORG"))
          s.append(z.tokens(j)) else s.append(y.tokens(j))
      result.append(Sentence(s))
    }
    result.toList
  }

  /**
    * Finds named entities of a text using bidirectional inference. Parts of entities are combined into
    * entire entities, for example, [B-ORG I-ORG I-ORG] patterns are combined into a single ORG.
    * @param sentences
    * @return a list of processing results, each result is a list of entities: (content, type), for example ("UBND TP.HCM", "ORG")
    */
  def run(sentences: List[Sentence]): List[List[(String, String)]] = {
    val output = combine(sentences)
    output.map(extract(_))
  }

  /**
    * Extracts named entities from a sentence.
    * @param s
    * @return a list of entities [(content, type)], for example [("Le Hong Phuong", "PER")]
    */
  def extract(s: Sentence): List[(String, String)] = {
    val tokens = s.tokens
    val xs = s.tokens.map(_.namedEntity).zipWithIndex
    val positions = xs.filter(p => p._1.startsWith("B-"))
    val entities = ListBuffer[(String, String)]()
    for (i <- positions.indices) {
      val kind = positions(i)._1.substring(2)
      val entity = new StringBuilder()
      val p = positions(i)._2
      entity.append(tokens(p).word)
      var j = p+1
      while (j < tokens.size && xs(j)._1 != "O") {
        entity.append(" ")
        entity.append(tokens(j).word)
        j = j + 1
      }
      entities.append((entity.toString.trim, kind))
    }
    entities.toList
  }

  /**
   * Infers NE tags for raw sentences. We first run a part-of-speech tagger and then run the name tagger.
   * @param xs a sequence of input raw sentences.
   * @return an annotated sentence with NE tags.
   */
  def inference(xs: List[String]): List[Sentence] = {
    def convertPoS(pos: String): String = {
      pos match {
        case "PUNCT" => "CH"
        case "Np" => "NNP"
        case _ => pos
      }
    }
    val ts = partOfSpeechTagger.tag(partOfSpeechModel, xs)
    val sentences = ts.map { t => 
      val tokens = t.map(pair => Token(pair._1, Map(Label.PartOfSpeech -> convertPoS(pair._2)))).toList
      Sentence(tokens.to[ListBuffer])
    }
    combine(sentences.toList)
  }

}

object Tagger {
  val logger: slf4j.Logger = LoggerFactory.getLogger(Tagger.getClass.getName)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val parser = new OptionParser[ConfigNER]("vlp.ner") {
      head("vlp.ner", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test/tag")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[Unit]('r', "reversed").action((_, conf) => conf.copy(reversed = true)).text("reversed mode")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language, either vie or eng")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("training data path")
      opt[Int]('u', "dimension").action((x, conf) => conf.copy(numFeatures = x)).text("number of features or domain dimension")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/ner/'")
      opt[String]('i', "input").action((x, conf) => conf.copy(input = x)).text("input path")
      opt[Unit]('j', "twoColumns").action((x, conf) => conf.copy(twoColumns = true)).text("two-column mode")
    }

    parser.parse(args, ConfigNER()) match {
      case Some(config) =>
        val sparkSession = SparkSession.builder().appName(getClass.getName).master(config.master).getOrCreate()
        val tagger = new Tagger(sparkSession, config)
        val sentences = CorpusReader.readCoNLL(config.dataPath, config.twoColumns)
        config.mode match {
          case "train" => tagger.train(sentences)
          case "test" => tagger.test(sentences)
          case "eval" => tagger.combine(config.input, config.input + ".out")
          case "tag" => 
            val xs = Source.fromFile(config.input, "UTF-8").getLines().toList.map(_.trim()).filter(_.nonEmpty)
            val ss = tagger.inference(xs)
            val lines = ss.map(s => {
              s.tokens.map(token => token.word + "/" + token.partOfSpeech + "/" + token.namedEntity).mkString(" ")
            })
            import scala.collection.JavaConversions._
            Files.write(Paths.get(config.input + ".out"), lines.toList, StandardCharsets.UTF_8, StandardOpenOption.CREATE)
            logger.info("Done.")
        }
        sparkSession.stop()
      case None =>
    }
  }
}

