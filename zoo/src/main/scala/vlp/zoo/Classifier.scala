package vlp.zoo

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}
import org.slf4j.LoggerFactory

import com.intel.analytics.bigdl.optim.{Adam, Top1Accuracy, Top5Accuracy}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.feature.text.{TextFeature, TextSet}
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import scopt.OptionParser
import com.intel.analytics.zoo.pipeline.api.keras.metrics.SparseCategoricalAccuracy
import java.text.DateFormat
import java.text.SimpleDateFormat
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.sql.RowFactory

import vlp.tok.TokenizerTransformer
import com.intel.analytics.zoo.pipeline.api.keras.metrics.Accuracy

import org.json4s._
import org.json4s.jackson.Serialization._
import org.json4s.jackson.Serialization
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import scala.io.Source
import java.nio.charset.StandardCharsets
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.Sample

/**
  * Configuration parameters of a Neural Network Classifier.
  *
  * @param master Spark master
  * @param mode train/eval/predict
  * @param executorMemory executor memory
  * @param dataPath path to the corpus
  * @param embeddingPath path to a pre-trained word embeddings file
  * @param modelPath pat to save trained model into
  * @param numFeatures number of most frequent tokens sorted by their frequency
  * @param encoder the encoder for the input sequence (cnn, gru, lstm)
  * @param encoderOutputDimension
  * @param maxSequenceLength the length of a sequence
  * @param batchSize batch size
  * @param epochs number of training epochs
  * @param learningRate learning rate
  * @param percentage percentage of the training data set to use (for quick experimentation)
  * @param partitions number of data partitions for Spark
  * @param minFrequency minimal frequency of tokens to be retained in training
  */
case class ConfigClassifier(
  master: String = "local[*]",
  mode: String = "eval",
  executorMemory: String = "8g",
  dataPath: String = "/opt/data/vne/5cats.utf8/",
  embeddingPath: String = "/opt/data/emb/vie/glove.6B.100d.txt",
  modelPath: String = "dat/zoo/tcl", 
  numFeatures: Int = 65536,
  encoder: String = "cnn",
  encoderOutputDimension: Int = 256,
  maxSequenceLength: Int = 256,
  batchSize: Int = 64,
  epochs: Int = 30,
  learningRate: Double = 0.001,
  percentage: Double = 1.0,
  partitions: Int = 4,
  minFrequency: Int = 2,
  verbose: Boolean = false,
  inputCol: String = "text",
  classCol: String = "category"
)

// SHINRA-related stuffs
case class ENE(ENE_id: String, ENE_name: String, score: Double = 1.0)
case class Result(
  pageid: String,
  title: String,
  lang: String = "vi",
  ENEs: List[ENE]
)

/**
  * Deep learning based text classifier. We can use CNN, LSTM or GRU architecture.
  *
  * @param sparkContext
  */
class Classifier(val sparkContext: SparkContext, val config: ConfigClassifier) {
  final val logger = LoggerFactory.getLogger(getClass.getName)
  val sparkSession = SparkSession.builder().getOrCreate()
  import sparkSession.implicits._
  implicit val formats = Serialization.formats(NoTypeHints)
  
  /**
    * Trains a neural text classifier on a text set and validate on a validation set. Trained model and word index are saved to
    * an external path specified in the configuration.
    * @param trainingSet
    * @param validationSet
    */
  def train(trainingSet: TextSet, validationSet: TextSet): TextClassifier[Float] = {
    logger.info("Processing text data set...")
    val transformedTrainingSet = trainingSet.tokenize()
      .word2idx(10, maxWordsNum = config.numFeatures, minFreq = config.minFrequency)
      .shapeSequence(config.maxSequenceLength)
      .generateSample()
    val wordIndex = transformedTrainingSet.getWordIndex
    transformedTrainingSet.saveWordIndex(config.modelPath + "/" + config.encoder + ".dict.txt")
    logger.info("Word index created.")

    val classifier = TextClassifier(Classifier.numLabels, config.embeddingPath, wordIndex, config.maxSequenceLength, config.encoder, config.encoderOutputDimension)
    val date = new SimpleDateFormat("yyyy-MM-dd.HHmmss").format(new java.util.Date())
    classifier.setTensorBoard(logDir = "/tmp/zoo/tcl", appName = config.encoder + "/" + date)
    classifier.compile(
      optimizer = new Adam(learningRate = config.learningRate),
      loss = SparseCategoricalCrossEntropy[Float](),
      metrics = List(new SparseCategoricalAccuracy[Float]())
    )
    logger.info("Preparing validation set...")
    val transformedValidationSet = validationSet.tokenize()
      .setWordIndex(wordIndex).word2idx()
      .shapeSequence(config.maxSequenceLength)
      .generateSample()

    classifier.fit(transformedTrainingSet, batchSize = config.batchSize, nbEpoch = config.epochs, transformedValidationSet)

    classifier.saveModel(config.modelPath + "/" + config.encoder + ".bin", overWrite = true)
    logger.info("Finish training model and saving word dictionary.")
    classifier
  }

  def predict(textSet: TextSet, classifier: TextClassifier[Float], docIds: Array[String] = Array.empty[String], outputFile: String = ""): TextSet = {
    val transformedTextSet = textSet.tokenize().loadWordIndex(config.modelPath + "/" + config.encoder + ".dict.txt").word2idx()
      .shapeSequence(config.maxSequenceLength).generateSample()
      DataSet
      TextFeature
    val result = classifier.predict(transformedTextSet, batchPerThread = config.partitions)
    val predictedClasses = classifier.predictClasses(transformedTextSet.toDistributed().rdd.map(_.getSample))
    if (outputFile.nonEmpty) {
      // restore the label map from an external JSON file
      val labelPath = config.modelPath + "/" + config.encoder + ".labels.json"
      implicit val formats = Serialization.formats(NoTypeHints)
      import scala.collection.JavaConversions._
      val mapSt = Files.readAllLines(Paths.get(labelPath))(0)
      val objectMapper = new ObjectMapper() with ScalaObjectMapper
      objectMapper.registerModule(DefaultScalaModule)
      val labels = objectMapper.readValue(mapSt, classOf[Map[String, Int]])
      val clazzMap = labels.keySet.map(key => (labels(key), key)).toMap[Int, String]
      val prediction = predictedClasses.collect().map(k => clazzMap(k))
      val output = docIds.zip(prediction).map(pair => Result(pair._1, "", "vi", List(ENE(pair._2, "", 0.0))))
            .map(result => Serialization.write(result))
      import scala.collection.JavaConversions._  
      Files.write(Paths.get(outputFile), output.toList, StandardCharsets.UTF_8)
    }
    result
  }

  def predict(texts: Seq[String], classifier: TextClassifier[Float]): TextSet = {
    val textRDD = sparkContext.parallelize(texts).map(content => TextFeature(content, 0))
    val textSet = TextSet.rdd(textRDD)
    predict(textSet, classifier)
  }

}

object Classifier {
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  var numLabels = 0

  /**
    * Reads a corpus in JSONL format, extract 'text' and 'category' fields to build a text set.
    *
    * @param sparkSession
    * @param config
    * @return a data frame of two columns (category, text)
    */
  def readJsonData(sparkSession: SparkSession, config: ConfigClassifier): TextSet = {
    val input = sparkSession.read.json(config.dataPath)
    val textSet = if (config.percentage < 1.0) input.sample(config.percentage) else input

    import sparkSession.implicits._
    val categories = textSet.select(config.classCol).flatMap(row => row.getString(0).split(",")).distinct.collect().sorted
    val labelPath = config.modelPath + "/" + config.encoder + ".labels.json"
    implicit val formats = Serialization.formats(NoTypeHints)
    val labels = if (config.mode == "train") {
      val map = categories.zipWithIndex.toMap
      numLabels = map.size
      println(s"Found ${numLabels} classes")
      println(map.mkString(", "))
      // write label map to an external JSON file
      val mapSt = Serialization.write(map)
      Files.write(Paths.get(labelPath), mapSt.getBytes(), StandardOpenOption.CREATE)
      map
    } else {
      // restore the label map from an external JSON file
      import scala.collection.JavaConversions._
      val mapSt = Files.readAllLines(Paths.get(labelPath))(0)
      val objectMapper = new ObjectMapper() with ScalaObjectMapper
      objectMapper.registerModule(DefaultScalaModule)
      objectMapper.readValue(mapSt, classOf[Map[String, Int]])
    }
    println("Creating text set. Please wait...")
    val textRDD = textSet.select(config.classCol, config.inputCol).rdd.map(row => {
      val content = row.getString(1).toLowerCase().split("\\s+").toArray.filter(w => w != ")" && w != "(" && w != "-")
      val text = content.map(token => TokenizerTransformer.convertNum(token))
      val label = row.getString(0).split(",").head
      TextFeature(text.mkString(" "), labels(label))
      }
    )
    TextSet.rdd(textRDD)
  }

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[ConfigClassifier]("zoo.tcl") {
      head("vlp.zoo.Classifier", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/predict")
      opt[String]('e', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("data path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/zoo/tcl/'")
      opt[String]('t', "encoder").action((x, conf) => conf.copy(encoder = x)).text("type of encoder, either cnn, lstm or gru")
      opt[Int]('o', "encoderOutputDimension").action((x, conf) => conf.copy(encoderOutputDimension = x)).text("output dimension of the encoder")
      opt[Int]('l', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("maximum sequence length for a text")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('n', "percentage").action((x, conf) => conf.copy(percentage = x)).text("percentage of the training set to use, default is 1.0")
      opt[String]('x', "inputCol").action((x, conf) => conf.copy(inputCol = x)).text("input column")
      opt[String]('y', "classCol").action((x, conf) => conf.copy(classCol = x)).text("class column")

    }
    parser.parse(args, ConfigClassifier()) match {
      case Some(config) =>
      val sparkConfig = Engine.createSparkConf()
        .setMaster(config.master)
        .set("spark.executor.memory", config.executorMemory)
        .setAppName("Neural Text Classifier")
      val sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate()
      val sparkContext = sparkSession.sparkContext
      Engine.init

      val app = new Classifier(sparkSession.sparkContext, config)
      val validationMethods = Array(new Accuracy[Float]())

      config.mode match {
        case "train" =>
          val textSet = readJsonData(sparkSession, config).toDistributed(sparkContext, config.partitions)
          val Array(trainingSet, validationSet) = textSet.randomSplit(Array(0.8, 0.2))
          app.train(trainingSet, validationSet)
        case "eval" =>
          val classifier = TextClassifier.loadModel[Float](config.modelPath + "/" + config.encoder + ".bin")
          classifier.setEvaluateStatus()
          val textSet = readJsonData(sparkSession, config).toDistributed(sparkContext, config.partitions)
          val prediction = app.predict(textSet, classifier)
          var accuracy = classifier.evaluate(prediction.toDistributed().rdd.map(_.getSample), validationMethods)
          println("validation accuracy = " + accuracy.mkString(", "))
        case "shinra" =>
          import sparkSession.implicits._
          val df = sparkSession.read.text("dat/shi/vi.txt").rdd.filter(row => row.getString(0).trim.nonEmpty).map(row => {
            val parts = (row.getString(0) + "\tNA").split("\t")
            RowFactory.create(parts: _*)
          })
          val schema = StructType(Array(StructField("pageid", StringType, false), StructField("text", StringType, false), 
            StructField("title", StringType, false), StructField("category", StringType, false), StructField("outgoingLink", StringType, false),
            StructField("redirect", StringType, false), StructField("clazz", StringType, false)))
          val ds = sparkSession.createDataFrame(df, schema)
          val docIds = ds.select("pageid").map(row => row.getString(0)).collect()
          val tokenizer = new TokenizerTransformer().setInputCol("text").setOutputCol("body").setConvertNumber(true).setToLowercase(true)
          val xs = tokenizer.transform(ds)
          xs.show()
          val textRDD = xs.select("body").rdd.map(row => {
            val content = row.getString(0).split("\\s+").toArray
            TextFeature(content.mkString(" "), -1)
          })
          val classifier = TextClassifier.loadModel[Float](config.modelPath + "/" + config.encoder + ".bin")
          classifier.setEvaluateStatus()
          val textSet = TextSet.rdd(textRDD)
          app.predict(textSet, classifier, docIds, "dat/shi/vi.json." + config.encoder)
      }
      sparkSession.stop()
      case None =>
    }
  }
}