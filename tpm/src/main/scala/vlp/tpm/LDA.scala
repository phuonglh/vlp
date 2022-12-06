package vlp.tpm

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.LDAModel
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import org.slf4j.LoggerFactory
import scopt.OptionParser
import org.apache.spark.rdd.RDD
import vlp.tok.SentenceDetection
import vlp.tok.Tokenizer

/**
  * phuonglh, 5/28/18, 23:39
  */
class LDA(val sparkContext: SparkContext, val config: ConfigLDA) {
  final val logger = LoggerFactory.getLogger(getClass.getName)
  val sparkSession = SparkSession.builder().getOrCreate()
  import sparkSession.implicits._

  def createDataset(jsonPath: String): Dataset[Document] = {
    val data = SparkSession.getActiveSession.get.read.json(jsonPath).as[Document]
    if (config.verbose)
      data.show()
    data
  }

  def train(dataset: Dataset[Document]): PipelineModel = {
    dataset.cache()
    dataset.show(true)
    // create pipeline
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setMinTokenLength(2)
    val stopWordsRemover = new StopWordsRemover().setInputCol("tokens").setOutputCol("words").setStopWords(StopWords.punctuations)
    val converter = new Converter().setInputCol("words").setOutputCol("ws")
    val unigramCounter = new CountVectorizer().setInputCol("ws").setOutputCol("us").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)
    val lda = new org.apache.spark.ml.clustering.LDA().setFeaturesCol("us").setK(config.k).setMaxIter(config.iterations)
    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, converter, unigramCounter, lda))
    logger.info("Training process started. Please wait...")
    logger.info("#(documents) = " + dataset.count())
    val model = pipeline.fit(dataset)
    model.write.overwrite().save(config.modelPath)
    model
  }
  
  def eval(dataset: Dataset[Document]): Unit = {
    val model = PipelineModel.load(config.modelPath)
    val vocabulary = model.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info("#(vocabulary) = " + vocabulary.size)
    if (config.verbose) vocabulary.foreach(logger.info(_))
    val output = model.transform(dataset)
    output.select("ws", "topicDistribution").show(10)
    val lda = model.stages(model.stages.size - 1).asInstanceOf[LDAModel]
    val ll = lda.logLikelihood(output)
    logger.info("The lower bound on the log likelihood of the corpus: " + ll)
    val topics = lda.describeTopics(config.top)
    logger.info("The topics described by their top-weighted terms: ")
    topics.show(config.k)
    val termIndexToString = new TermIndexToString(vocabulary).setInputCol("termIndices").setOutputCol("terms")
    termIndexToString.transform(topics).select("topic", "terms").show(config.k, false)
  }

  /**
    * Predicts the topic distribution of a word-segmented document.
    * @param document
    * @param model
    * @return a probability distribution for k topics
    */
  def predict(document: Document, model: PipelineModel): Map[String, Double] = {
    import sparkSession.implicits._
    val dataset = sparkSession.sparkContext.parallelize(List(document)).toDF("text").as[Document]
    val output = model.transform(dataset)
    val first = output.select("topicDistribution").head()
    val topicValues = first.getAs[DenseVector](0).values.zipWithIndex
    val topicMap = topicValues.map(pair => (pair._2.toString, pair._1)).toMap
    topicMap
  }

  /**
    * Lists top terms of each topic, returns a list of k term sequences, each sequence has top terms.
    * @param model
    * @return a sequence of strings
    */
  def topicTerms(model: PipelineModel): Seq[String] = {
    val vocabulary = model.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    val lda = model.stages(model.stages.size - 1).asInstanceOf[LDAModel]
    val topics = lda.describeTopics(config.top)
    val termIndexToString = new TermIndexToString(vocabulary).setInputCol("termIndices").setOutputCol("terms")
    val terms = termIndexToString.transform(topics).select("terms").collect()
    terms.map(_.getAs[Seq[String]](0).mkString(" "))
  }
}

object LDA {

  final val logger = LoggerFactory.getLogger(getClass.getName)
  final val tokenizer = new Tokenizer()

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val parser = new OptionParser[ConfigLDA]("vlp.tpm") {
      head("vlp.tpm", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[String]('e', "executorMemory").action((x, conf) => conf.copy(memory = x)).text("executor memory, default is 8g")
      opt[Unit]('v', "verbose").action((x, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[Int]('k', "topics").action((x, conf) => conf.copy(k = x)).text("number of topics")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("data path, default is 'dat/txt/fin.json'")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/tpm/'")
    }
    parser.parse(args, ConfigLDA()) match {
      case Some(config) => {
        val sparkSession = SparkSession.builder().appName(getClass.getName).master(config.master)
          .config("spark.executor.memory", config.memory)
          .config("spark.driver.host", "localhost")
          .getOrCreate()
        implicit val formats = Serialization.formats(NoTypeHints)
        logger.info(Serialization.writePretty(config))
        val lda = new LDA(sparkSession.sparkContext, config)
        config.mode match {
          case "train" => {
            val dataset = lda.createDataset(config.dataPath)
            lda.train(dataset)
            lda.eval(dataset)
          }
          case "eval" => {
            val dataset = lda.createDataset(config.dataPath)
            lda.eval(dataset)
          }
          case "predict" =>
            val model = PipelineModel.load(config.modelPath)
            logger.info(lda.topicTerms(model).toString())
            val document = Document("Do_vậy , Chủ_tịch FPT Software khẳng_định năng_suất lao_động là yếu_tố sống_còn của các doanh_nghiệp . Tuy_nhiên , việc nhân_viên \" chăm_chỉ hơn mỗi ngày  nhiệt_tình hơn mỗi giờ \" + " +
              "thực_tế không giải_quyết được bao_nhiêu . Điều quan_trọng , giúp biến_đổi về \" chất \" , là phải ứng_dụng chuyển_đổi số .")
            val result = lda.predict(document, model)
            logger.info(result.toString)
        }
        sparkSession.stop()
      }
      case None => {}
    }
  }
}
