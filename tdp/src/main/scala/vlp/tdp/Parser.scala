package vlp.tdp

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scopt.OptionParser

/**
  * Created by phuonglh on 6/24/17.
  * 
  * A transition-based dependency parser.
  * 
  */
class Parser(spark: SparkSession, configTDP: ConfigTDP, classifierType: ClassifierType.Value, useSuperTag: Boolean = false) extends Serializable {
  var verbose: Boolean = false
  val logger = LoggerFactory.getLogger(getClass)
  val pipeline = classifierType match {
    case ClassifierType.MLR => PipelineModel.load(configTDP.modelPath + configTDP.language + "/mlr")
    case ClassifierType.MLP => PipelineModel.load(configTDP.modelPath + configTDP.language + "/mlp")
  }

  val featureExtractor = if (classifierType == ClassifierType.MLR) new FeatureExtractor(true, useSuperTag) ; else new FeatureExtractor(true, useSuperTag) 
  
  val model = classifierType match {
    case ClassifierType.MLR => new MLR(spark, pipeline, featureExtractor)
    case ClassifierType.MLP => new MLP(spark, pipeline, featureExtractor)
  } 

  def setVerbose(verbose: Boolean): Unit = this.verbose = verbose
  /**
    * Parses a sentence and returns a dependency graph.
    *
    * @param sentence a sentence
    * @return a dependency graph
    */
  def parse(sentence: Sentence): Graph = {
    val stack = mutable.Stack[String]()
    val queue = mutable.Queue[String]()
    // create input tokens without head and dependencyLabel information
    val x = sentence.tokens.map{ t => Token(t.word, mutable.Map[Label.Value, String](
      Label.Id -> t.id,
      Label.Lemma -> t.lemma,
      Label.UniversalPartOfSpeech -> t.universalPartOfSpeech,
      Label.PartOfSpeech -> t.partOfSpeech,
      Label.FeatureStructure -> t.featureStructure
    ))}
    val s = Sentence(x)
    // transition-based dependency parsing
    s.tokens.foreach(t => queue.enqueue(t.id))

    val arcs = new ListBuffer[Dependency]()
    val easyFirstAnnotator = new EasyFirstAnnotator(configTDP.language)
    arcs ++= easyFirstAnnotator.annotate(sentence)
    
    var config = Config(s, stack, queue, arcs).next("SH")

    while (!config.isFinal) {
      val best = model.predict(config)
      val transition = best.head
      transition match {
        case "RE" => 
          if (!config.isReducible) 
            if (verbose) logger.warn("Wrong RE label for config: " + config.toPrettyString())
        case _ =>
      }
      config = config.next(transition)
    }
    // annotate the tokens using the dependency edges
    for (dependency <- arcs) {
      val token = s.token(dependency.dependent)
      token.annotation += Label.Head -> dependency.head
      token.annotation += Label.DependencyLabel -> dependency.label
    }
    Graph(s)
  }

  /**
    * Parses a list of sentences and returns a list of dependency graphs.
    * @param sentences a list of sentences
    * @return a list of dependency graphs
    */
  def parse(sentences: List[Sentence]): List[Graph] = {
    val df = spark.sparkContext.parallelize(sentences)
    df.map(sentence => parse(sentence)).collect().toList
  }

  /**
    * Evaluates the accuracy of the parser on a list of graphs.
    * @param graphs
    * @param countPuncts
    * @return a pair of (uas, las) scores.
    */
  def eval(graphs: List[Graph], countPuncts: Boolean = false): (Double, Double) = {
    Evaluation.eval(this, graphs, countPuncts)
    (Evaluation.uas, Evaluation.las)
  }
  
  def info(): Unit = {
    logger.info(model.info())
  }

  /**
    * Parses a raw string given part-of-speech information in the form of "w1/up1/p1 w2/up2/p2 ... ",
    * where "w" are words, "up" are universal part-of-speech tags and "p" are local part-of-speech tags.
    *
    * @param sentence
    * @return a graph
    */
  def parseWithPartOfSpeech(sentence: String): Graph = {
    val xs = sentence.split("""\s+""").toList
    val tokens = xs.zipWithIndex.map { case (x, id) =>
      val parts = x.split("/")
      val word = if (parts(0).nonEmpty) parts(0) else "/"
      val annotation = if (parts.size == 4) { // that is: "/PUNCT/PUNCT"
        mutable.Map[Label.Value, String](Label.Id -> (id + 1).toString(), Label.Lemma -> word.toLowerCase(), 
          Label.UniversalPartOfSpeech -> parts(2),
          Label.PartOfSpeech -> parts(3))
      } else if (parts.size == 3) {
        mutable.Map[Label.Value, String](Label.Id -> (id + 1).toString(), Label.Lemma -> word.toLowerCase(), 
          Label.UniversalPartOfSpeech -> parts(1),
          Label.PartOfSpeech -> parts(2))
      } else if (parts.size == 2) {
        mutable.Map[Label.Value, String](Label.Id -> (id + 1).toString(), Label.Lemma -> word.toLowerCase(),
          Label.UniversalPartOfSpeech -> parts(1))
      } else {
        mutable.Map[Label.Value, String](Label.Id -> (id + 1).toString(), Label.Lemma -> word.toLowerCase())
      }
      Token(word, annotation)
    }
    val s = Sentence(GraphReader.root +: tokens.to[ListBuffer])
    parse(s)
  }
}

/**
  * The companion object of the Parser class.
  */
object Parser {
  val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val optionParser = new OptionParser[ConfigTDP]("vlp.tdp.Parser") {
      head("vlp.tdp.Parser", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/debug")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/tdp/'")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[String]('c', "classifier").action((x, conf) => conf.copy(classifier = x)).text("classifier, either mlr or mlp")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language, either vie or eng, default is vie")
      opt[Unit]('x', "extended").action((_, conf) => conf.copy(extended = true)).text("extended mode for English parsing")
    }

    optionParser.parse(args, ConfigTDP()) match {
      case Some(config) =>
        val spark = SparkSession.builder().appName(getClass.getName)
        .master(config.master)
        .config("spark.executor.memory", config.memory)
        .config("spark.driver.host", "localhost")
        .getOrCreate()
      val corpusPack = if (config.language == "eng") new CorpusPack(Language.English) ; else new CorpusPack()
      val (trainingGraphs, developmentGraphs) = if (corpusPack.dataPaths._1 != corpusPack.dataPaths._2) 
        (GraphReader.read(corpusPack.dataPaths._1), GraphReader.read(corpusPack.dataPaths._2))
        else {
          val graphs = GraphReader.read(corpusPack.dataPaths._1)
          scala.util.Random.setSeed(220712)
          val randomGraphs = scala.util.Random.shuffle(graphs)
          val trainingSize = (randomGraphs.size * 0.8).toInt
          (randomGraphs.take(trainingSize), randomGraphs.slice(trainingSize, randomGraphs.size))
        }
      val classifierType = config.classifier match {
        case "mlr" => ClassifierType.MLR
        case "mlp" => ClassifierType.MLP
      }
      val parser = new Parser(spark, config, classifierType, config.extended)
      parser.setVerbose(config.verbose)
      parser.info()
      config.mode match {
        case "eval" =>
          for (graphs <- List(developmentGraphs, trainingGraphs)) {
            val (uasP, lasP) = parser.eval(graphs, true)
            logger.info(s"uas = $uasP, las = $lasP")
          }
        case "test" =>
          val x = developmentGraphs.take(2).map(g => g.sentence)
          val y = parser.parse(x)
          for (i <- 0 until y.length) {
            logger.info("\n" + developmentGraphs(i).toString + "\n")
            logger.info("\n" + y(i).toString + "\n")
          }
        case "parse" => 
          val sentence = "Nên/SCONJ/SC trước_nhất/N/N người/N/Nc đảng_viên/N/N phải/VERB/V làm_gương/VERB/V ./PUNCT/PUNCT"
          val graph = parser.parseWithPartOfSpeech(sentence)
          logger.info(graph.toString)
      }
      spark.stop()
      case None => 
    }
  }
}
