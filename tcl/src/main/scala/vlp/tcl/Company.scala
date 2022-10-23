package vlp.tcl

import org.json4s.jackson.Serialization
import org.json4s._
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Dataset, DataFrame, RowFactory, SparkSession}
import org.apache.spark.sql.functions.lit
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import org.slf4j.LoggerFactory
import scopt.OptionParser
import java.io.File

/*
 * 
 * phuonglh, October 2022.
 * 
 */

case class Sample(
  ticker: String = "", 
  label: String = "", 
  description: String = ""
) {
  val cs = this.getClass().getConstructors()
  def fromSeq(xs: Array[String]) = cs(0).newInstance(xs: _*).asInstanceOf[Sample]
}

class Company(sparkSession: SparkSession, config: ConfigTCL) {
  final val logger = LoggerFactory.getLogger(getClass.getName)
  import sparkSession.implicits._

  def train(dataset: Dataset[Sample]): PipelineModel = {
    dataset.cache()
    // create pipeline
    val tokenizer = new RegexTokenizer().setInputCol(config.inputColumnName).setOutputCol("tokens").setPattern("""[\s,;.?:'")(-]+""")
    val stopWordsRemover = new StopWordsRemover().setInputCol("tokens").setOutputCol("unigrams").setStopWords(StopWords.punctuations)
    val unigramCounter = new CountVectorizer().setInputCol("unigrams").setOutputCol("us").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)
    val labelIndexer = new StringIndexer().setInputCol("label").setHandleInvalid("skip").setOutputCol("clazz")
    val classifierType = config.classifier
    val pipeline = if (classifierType == "mlr") {
      val bigram = new NGram().setInputCol("unigrams").setOutputCol("bigrams").setN(2)
      val bigramCounter = new CountVectorizer().setInputCol("bigrams").setOutputCol("bs").setMinDF(config.minFrequency).setVocabSize(2*config.numFeatures)
      val assembler = new VectorAssembler().setInputCols(Array("us", "bs")).setOutputCol("features")
      val mlr = new LogisticRegression().setMaxIter(config.iterations).setRegParam(config.lambda).setStandardization(false).setLabelCol("clazz")
      new Pipeline().setStages(Array(labelIndexer, tokenizer, stopWordsRemover, unigramCounter, bigram, bigramCounter, assembler, mlr))
    } else if (classifierType == "mlp") {
      val featureHashing = new HashingTF().setInputCol("unigrams").setOutputCol("features").setNumFeatures(config.numFeatures).setBinary(true)
      val numLabels = labelIndexer.fit(dataset).labels.size
      logger.info(s"numLabels = ${numLabels}")
      val xs = config.hiddenUnits.trim
      val hiddenLayers = if (xs.nonEmpty) xs.split("[\\s,]+").map(_.toInt); else Array[Int]()
      val layers = Array(config.numFeatures) ++ hiddenLayers ++ Array[Int](numLabels)
      logger.info(layers.mkString(", "))
      val mlp = new MultilayerPerceptronClassifier().setMaxIter(config.iterations).setBlockSize(config.batchSize).setSeed(123).setLayers(layers).setLabelCol("clazz")
      new Pipeline().setStages(Array(labelIndexer, tokenizer, stopWordsRemover, featureHashing, mlp))
    } else if (classifierType == "rfc") {
      val featureHashing = new HashingTF().setInputCol("unigrams").setOutputCol("features").setNumFeatures(config.numFeatures).setBinary(false)
      val rfc = new RandomForestClassifier().setNumTrees(config.numTrees).setMaxDepth(config.maxDepth).setLabelCol("clazz")
      new Pipeline().setStages(Array(labelIndexer, tokenizer, stopWordsRemover, featureHashing, rfc))
    } else {
      logger.error("Not support classifier type: " + classifierType)
      new Pipeline()
    }
    dataset.show(10)
    logger.info("#(documents) = " + dataset.count())
    logger.info("Training process started. Please wait...")
    val model = pipeline.fit(dataset)
    model.write.overwrite().save(config.modelPath + "/" + classifierType)
    model
  }

  def eval(model: PipelineModel, dataset: Dataset[Sample]): Unit = {
    val transformer = model.stages(3)
    if (transformer.isInstanceOf[CountVectorizerModel]) {
      val vocabulary = transformer.asInstanceOf[CountVectorizerModel].vocabulary
      logger.info("#(vocabulary) = " + vocabulary.size)
    } else if (transformer.isInstanceOf[HashingTF]) {
        val numFeatures = transformer.asInstanceOf[HashingTF].getNumFeatures
        logger.info("#(numFeatures) = " + numFeatures)
    } else logger.error(s"Error in reading information from ${transformer.getClass.getName}")

    dataset.show(10)
    val outputDF = model.transform(dataset)

    import sparkSession.implicits._
    val predictionAndLabels = outputDF.select("clazz", "prediction").map(row => (row.getDouble(0), row.getDouble(1))).rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val scores = (metrics.accuracy, metrics.weightedFMeasure)
    logger.info(s"scores = $scores")
    if (config.verbose) {
      outputDF.show(10)
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

  def eval(dataset: Dataset[Sample]): Unit = {
    val model = PipelineModel.load(config.modelPath + "/" + config.classifier.toLowerCase())
    eval(model, dataset)
  }

  def predict(dataset: Dataset[Sample], outputFile: String): Unit = {
    val model = PipelineModel.load(config.modelPath + "/" + config.classifier.toLowerCase())
    val labels = model.stages(0).asInstanceOf[StringIndexerModel].labels
    val outputDF = model.transform(dataset)
    import sparkSession.implicits._
    implicit val formats = Serialization.formats(NoTypeHints)
    val result = outputDF.select("ticker", "prediction", "probability")
      .map(row => (row.getString(0), labels(row.getDouble(1).toInt), row.getAs[DenseVector](2)))
      .collect()
      .map(result => Serialization.write(result))
    import scala.collection.JavaConversions._  
    Files.write(Paths.get(outputFile), result.toList, StandardCharsets.UTF_8)
  }

  def predict(inputFile: String, outputFile: String): Unit = {
    import sparkSession.implicits._
    val dataset = sparkSession.read.options(Map("header" -> "true")).csv(inputFile).as[Sample]
    logger.info("Number of samples for prediction = " + dataset.count())
    predict(dataset, outputFile)
  }

}

object Company {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val logger = LoggerFactory.getLogger(getClass.getName)

    val parser = new OptionParser[ConfigTCL]("vlp.tcl") {
      head("vlp.tcl", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[String]('c', "classifier").action((x, conf) => conf.copy(classifier = x)).text("classifier, either mlr/mlp/rfc")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[String]('h', "hiddenUnits").action((x, conf) => conf.copy(hiddenUnits = x)).text("hidden units in MLP")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[Int]('t', "numTrees").action((x, conf) => conf.copy(numTrees = x)).text("number of trees if using RFC, default is 256")
      opt[Int]('e', "maxDepth").action((x, conf) => conf.copy(maxDepth = x)).text("max tree depth if using RFC, default is 15")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("data path")
      opt[Double]('n', "percentage").action((x, conf) => conf.copy(percentage = x)).text("percentage of the training data to use, default is 1.0")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/tcl/'")
      opt[String]('j', "inputColumnName").action((x, conf) => conf.copy(inputColumnName = x)).text("input column name")
      opt[Int]('l', "maxTokenLength").action((x, conf) => conf.copy(maxTokenLength = x)).text("max token length for a text, default is 50")
      opt[String]('i', "input").action((x, conf) => conf.copy(input = x)).text("input path")
      opt[String]('o', "output").action((x, conf) => conf.copy(output = x)).text("output path")
    }
    parser.parse(args, ConfigTCL()) match {
      case Some(config) =>
        val sparkSession = SparkSession.builder().appName("tcl.Company").master(config.master).getOrCreate()
        import sparkSession.implicits._
        import org.apache.spark.sql.functions._
        implicit val formats = Serialization.formats(NoTypeHints)
        logger.info(Serialization.writePretty(config))
        val app = new Company(sparkSession, config)
        val dataset = sparkSession.read.options(Map("header" -> "true")).csv(config.dataPath).as[Sample]
          .filter(col("description").isNotNull)
        config.mode match {
          case "train" => {            
            val trainingSet = if (config.percentage < 1.0) dataset.sample(config.percentage) else dataset
            val model = app.train(trainingSet)
            app.eval(model, trainingSet)
          }
          case "eval" =>
            val model = PipelineModel.load(config.modelPath + "/" + config.classifier.toLowerCase())
            app.eval(model, dataset)
          case "predict" => app.predict(config.input, config.output)
        }
        sparkSession.stop()
      case None =>
      }
  }
}
