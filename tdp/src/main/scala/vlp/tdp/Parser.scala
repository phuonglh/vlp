package vlp.tdp

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Created by phuonglh on 6/24/17.
  * 
  * A transition-based dependency parser.
  * 
  */
class Parser(spark: SparkSession, corpusPack: CorpusPack, classifierType: ClassifierType.Value) extends Serializable {
  var verbose: Boolean = false
  val logger = LoggerFactory.getLogger(getClass)
  val modelPath = corpusPack.modelPath
  val pipeline = classifierType match {
    case ClassifierType.MLR => PipelineModel.load(modelPath + "/mlr")
    case ClassifierType.MLP => PipelineModel.load(modelPath + "/mlp")
  }
  val featureExtractor = if (classifierType == ClassifierType.MLR) new FeatureExtractor(true) ; else new FeatureExtractor(false) 
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
    val stack = new mutable.Stack[String]()
    val queue = new mutable.Queue[String]()
    // create input tokens without head and dependencyLabel information
    val x = sentence.tokens.map(t => Token(t.word, mutable.Map[Label.Value, String](
      Label.Id -> t.id,
      Label.Lemma -> t.lemma,
      Label.UniversalPartOfSpeech -> t.universalPartOfSpeech,
      Label.PartOfSpeech -> t.partOfSpeech,
      Label.FeatureStructure -> t.featureStructure
    )))
    val s = Sentence(x)
    // transition-based dependency parsing
    s.tokens.foreach(t => queue.enqueue(t.id))
    stack.push(queue.dequeue())
    val arcs = new ListBuffer[Dependency]()
    var config = Config(s, stack, queue, arcs)
    
    // easy-first annotation
    val easyArcs = new EasyFirstAnnotator(corpusPack.language).annotate(s)
    if (!easyArcs.isEmpty) logger.info(easyArcs.mkString(", "))
    arcs ++= easyArcs

    while (!config.isFinal) {
      val best = model.predict(config)
      val transition = best.head
      transition match {
        case "SH" => config = config.next(transition)
        case tr if tr.startsWith("RA") => {
          config = config.next(transition)
          val dependency = config.arcs.last
          val token = s.token(dependency.dependent)
          token.annotation += (Label.Head -> dependency.head, Label.DependencyLabel -> dependency.label)
        }
        case tr if (tr.startsWith("LA")) => {
          // check precondition for LA transition: the top element on stack is not ROOT
          if (config.stack.top != "0") {
            if (!config.isReducible && verbose) {
              logger.warn("Impossible " + transition + " for " + config.words + "; " + config.stack + "; " + config.queue + "; " + config.arcs)
            }
            config = config.next(transition)
            val dependency = config.arcs.last
            val token = s.token(dependency.dependent)
            token.annotation += (Label.Head -> dependency.head, Label.DependencyLabel -> dependency.label)
          } else {
            config = config.next("SH")
          }
        }
        case "RE" => {
          if (config.isReducible) config = config.next("RE"); else {
            if (verbose)
              logger.warn("Impossible RE: " + config.words + "; " + config.stack + "; " + config.queue + "; " + config.arcs)
            if (best.length > 1) config = config.next(best.last) ; else config = config.next("SH")
          }
        }
      }
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
    * @return a pair of (uas, las) scores.
    */
  def eval(graphs: List[Graph]): (Double, Double) = {
    Evaluation.eval(this, graphs)
    (Evaluation.uas, Evaluation.las)
  }
  
  def info(): Unit = {
    logger.info(model.info())
  }
}

/**
  * The companion object of the Parser class.
  */
object Parser {
  val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val config: ConfigTDP = ConfigTDP(verbose = true)
    val spark = SparkSession.builder().appName(getClass.getName)
      .master(config.master).config("spark.executor.memory", config.memory)
      .getOrCreate()
    val corpusPack = if (config.language == "eng") new CorpusPack(Language.English) ; else new CorpusPack()
    val graphs = GraphReader.read(corpusPack.dataPaths._2)
    val classifierType = config.classifier match {
      case "mlr" => ClassifierType.MLR
      case "mlp" => ClassifierType.MLP
    }
    val parser = new Parser(spark, corpusPack, classifierType)
    parser.setVerbose(config.verbose)
    parser.info()
    config.mode match {
      case "eval" =>
        val (uas, las) = parser.eval(graphs)
        logger.info(s"uas = $uas, las = $las")
      case "test" =>
        val x = graphs.take(5).map(g => g.sentence)
        val y = parser.parse(x)
        for (i <- 0 until y.length) {
          logger.info("\n" + graphs(i).toString + "\n")
          logger.info("\n" + y(i).toString + "\n")
        }
    }
    spark.stop()
  }
}
