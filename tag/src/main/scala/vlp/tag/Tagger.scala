package vlp.tag

import vlp.tok.TokenizerTransformer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory
import scopt.OptionParser

import scala.io.Source

case class Datum(x: String, y: String)

case class ConfigPoS(
  master: String = "local[*]",
  mode: String = "eval",
  dataPath: String = "dat/vtb-tagged.txt",
  markovOrder: Int = 2,
  minDF: Int = 3,
  dimension: Int = 16384,
  modelPath: String = "dat/tag/",
  verbose: Boolean = false,
  input: String = "dat/input.txt"
)

/**
  * phuonglh, 10/11/17, 4:14 PM
  */
class Tagger(sparkSession: SparkSession, config: ConfigPoS) {
  final val logger = LoggerFactory.getLogger(getClass.getName)

  def readTextData(path: String, resource: Boolean = false): List[Datum] = {
    val input = if (resource)
      Source.fromInputStream(getClass.getResourceAsStream(path), "UTF-8")
    else
      Source.fromFile(path, "UTF-8")
    val lines = input.getLines().filterNot(_.trim.isEmpty).toList
    lines.map(line => {
      val tokens = line.split("\\s+")
      val pairs = tokens.map(token => {
        if (!token.contains("//")) {
          val j = token.lastIndexOf('/')
          (token.substring(0, j), token.substring(j+1))
        } else ("/", "/")
      })
      // convert the tokens and tags and make a data point
      val x = pairs.map(p => Tagger.convert(p._1)).mkString(" ")
      val y = pairs.map(p => Tagger.convert(p._2)).mkString(" ")
      Datum(x, y)
    })
  }

  /**
    * Trains a CMM and saves the model to an external files.
    * Parameters are specified in the options.
    * @return a pipeline model.
    */
  def train: PipelineModel = {
    import sparkSession.implicits._
    val data = readTextData(config.dataPath)
    val input = sparkSession.createDataFrame(data).as[Datum]
    input.cache()
    input.show(10)
    val wordTokenizer = new RegexTokenizer().setInputCol("x").setOutputCol("words").setToLowercase(false)
    val tagTokenizer = new RegexTokenizer().setInputCol("y").setOutputCol("tags").setToLowercase(false)
    val featureExtractor = new FeatureExtractor().setWordCol("words").setTagCol("tags").setOutputCol("samples").setMarkovOrder(config.markovOrder)
      .setFeatureTypes(Array("currentWord", "previousWord", "nextWord", "currentShape", "nextShape"))
    val sampleExtractor = new SampleExtractor().setSampleCol("samples").setTokenCol("word").setFeatureCol("f").setLabelCol("tag")
    val featureTokenizer = new Tokenizer().setInputCol("f").setOutputCol("tokens")
    val featureVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("features").setMinDF(config.minDF).setVocabSize(config.dimension)
    val tagIndexer = new StringIndexer().setInputCol("tag").setOutputCol("label").setHandleInvalid("skip")

    val pipeline = new Pipeline()
    val mlr = new LogisticRegression().setMaxIter(400)
    pipeline.setStages(Array(wordTokenizer, tagTokenizer, featureExtractor, sampleExtractor, featureTokenizer, featureVectorizer, tagIndexer, mlr))
    val model = pipeline.fit(input)
    model.write.overwrite().save(config.modelPath)
    val output = model.transform(input)
    if (config.verbose) {
      output.show(20, false)
      logger.info("Number of training samples = " + output.count())
      val vectorizer = model.stages.find(_.isInstanceOf[CountVectorizerModel]).get.asInstanceOf[CountVectorizerModel]
      logger.info("Number of dimensions = " + vectorizer.vocabulary.size)
    }
    model
  }

  /**
    * Evaluates the classification performance on a test set.
    */
  def evaluate: Unit = {
    val model = PipelineModel.load(config.modelPath)
    import sparkSession.implicits._
    val data = readTextData(config.dataPath)
    val input = sparkSession.createDataFrame(data).as[Datum]
    val output = model.transform(input)

    val predictionAndLabels = output.select("label", "prediction").map(row => (row.getDouble(0), row.getDouble(1))).rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val scores = (metrics.accuracy, metrics.weightedFMeasure)
    logger.info(s" (A, F) scores = $scores")

    if (config.verbose) {
      val labels = metrics.labels
      labels.foreach(label => {
        val sb = new StringBuilder()
        sb.append(s"Precision($label) = " + metrics.precision(label) + ", ")
        sb.append(s"Recall($label) = " + metrics.recall(label) + ", ")
        sb.append(s"F($label) = " + metrics.fMeasure(label))
        logger.info(sb.toString)
      })
    }
  }

  /**
    * Uses a pre-trained model to tag new sentences in the input.
    */
  def tag: Unit = {
    val model = PipelineModel.load(config.modelPath)
    val sentences = Source.fromFile(config.input, "UTF-8").getLines().filterNot(_.trim.isEmpty).toList
    val output = tag(model, sentences)
    output.foreach(s => println(s.mkString(" ")))
  }

  /**
    * Tags new sentences by using a pre-trained pipeline model.
    * @param model a pre-trained pipeline model.
    * @param sentences word sequences
    * @return a list of sequences, each sequence is a list of (word, tag) tuples.
    */
  def tag(model: PipelineModel, sentences: Seq[String]): List[List[(String, String)]] = {
    // create a CMM
    val vectorizer = model.stages.find(_.isInstanceOf[CountVectorizerModel]).get.asInstanceOf[CountVectorizerModel]
    val vocabulary = vectorizer.vocabulary.zip(0 until vectorizer.vocabulary.size).toMap
    val mlr = model.stages.find(_.isInstanceOf[LogisticRegressionModel]).get.asInstanceOf[LogisticRegressionModel]
    val weights = mlr.coefficientMatrix
    val intercept = mlr.interceptVector
    val labels = model.stages.find(_.isInstanceOf[StringIndexerModel]).get.asInstanceOf[StringIndexerModel].labels
    val featureExtractor = model.stages.find(_.isInstanceOf[FeatureExtractor]).get.asInstanceOf[FeatureExtractor]
    val featureTypes = featureExtractor.getFeatureTypes
    val markovOrder = featureExtractor.getMarkovOrder
    val cmm = new CMM(vocabulary, weights, intercept, labels, featureTypes, markovOrder)
    // tag the sentences
    tag(cmm, sentences, false)
  }

  /**
    * Tags new sentences by using a pre-trained CMM.
    * @param cmm a CMM
    * @param sentences word sequences
    * @param verbose
    * @return a list of tuples, each tuple is a pair (word, tag)
    */
  def tag(cmm: CMM, sentences: Seq[String], verbose: Boolean): List[List[(String, String)]] = {
    import sparkSession.implicits._
    val input = sentences.toDF("x")
    val wordTokenizer = new TokenizerTransformer().setInputCol("x").setOutputCol("words")
    val tokens = wordTokenizer.transform(input).select("words").rdd.map(row => row.getAs[String](0)).collect()
    val output = tokens.map(sentence => {
      val xs = sentence.split("\\s+")
      val ws = xs.map(Tagger.convert(_))
      val ts = Array.fill[String](ws.length)("NA")
      cmm.predict(ws, ts)
      xs.zip(ts)
    })
    output.map(_.toList).toList
  }
  
}

object Tagger {
  final val logger = LoggerFactory.getLogger(Tagger.getClass.getName)
  final val punctuations = Array(",", ".", ":", ";", "?", "!", "\"", "'", "/", "...", "-", "LBKT", "RBKT", "--", "``", "''", ")", "(")
  def convert(token: String): String = if (punctuations.contains(token)) "PUNCT" else token.replaceAll(",", ".")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val parser = new OptionParser[ConfigPoS]("vlp.tag") {
      head("vlp.tag", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/tag")
      opt[Int]('f', "minDF").action((x, conf) => conf.copy(minDF = x)).text("min feature frequency")
      opt[Int]('u', "dimension").action((x, conf) => conf.copy(dimension = x)).text("domain dimension")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("training path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path")
      opt[String]('i', "input").action((x, conf) => conf.copy(input = x)).text("input path")
      opt[Unit]('v', name="verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
    }
    parser.parse(args, ConfigPoS()) match {
      case Some(config) =>
        val sparkSession = SparkSession.builder().appName(getClass.getName).master(config.master).getOrCreate()
        logger.info(config.toString)
        val tagger = new Tagger(sparkSession, config)
        config.mode match {
          case "train" => {
            tagger.train
            tagger.evaluate
          }
          case "tag" => tagger.tag
          case "eval" => tagger.evaluate
          case _ => logger.error("Invalid mode " + config.mode + ". Use either 'train', 'tag' or 'eval'.")
        }
        sparkSession.stop()
      case None => {}
    }
  }
}
