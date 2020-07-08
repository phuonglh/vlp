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
import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.zoo.pipeline.api.keras.metrics.SparseCategoricalAccuracy
import java.text.DateFormat
import java.text.SimpleDateFormat
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.sql.RowFactory

import java.io.File

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
  * @param trainingSplit the split portion of the data for training
  * @param batchSize batch size
  * @param epochs number of training epochs
  * @param learningRate learning rate
  * @param partitions number of data partitions for Spark
  * @param minFrequency minimal frequency of tokens to be retained in training
  */
case class ConfigClassifier(
  master: String = "local[*]",
  mode: String = "eval",
  executorMemory: String = "8g",
  dataPath: String = "/opt/data/vne/5cats.utf8/",
  embeddingPath: String = "/opt/data/emb/vie/glove.6B.200d.txt",
  modelPath: String = "dat/zoo/tcl/", // need the last back slash
  numFeatures: Int = 16384,
  encoder: String = "cnn",
  encoderOutputDimension: Int = 256,
  maxSequenceLength: Int = 500,
  trainingSplit: Double = 0.8,
  batchSize: Int = 64,
  epochs: Int = 15,
  learningRate: Double = 0.001,
  partitions: Int = 4,
  minFrequency: Int = 2,
  verbose: Boolean = false
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
  
  /**
    * Trains a neural text classifier on a text set and validate on a validation set. Trained model and word index are saved to
    * an external path specified in the configuration.
    * @param trainingSet
    * @param validationSet
    */
  def train(trainingSet: TextSet, validationSet: TextSet): TextClassifier[Float] = {
    println("Processing text data set...")
    val transformedTrainingSet = trainingSet.tokenize()
      .word2idx(10, maxWordsNum = config.numFeatures, minFreq = config.minFrequency)
      .shapeSequence(config.maxSequenceLength)
      .generateSample()
    val wordIndex = transformedTrainingSet.getWordIndex
    transformedTrainingSet.saveWordIndex(config.modelPath + "/wordIndex.txt")

    val classifier = TextClassifier(Classifier.numLabels, config.embeddingPath, wordIndex, config.maxSequenceLength, config.encoder, config.encoderOutputDimension)
    val date = new SimpleDateFormat("yyyy-MM-dd.HHmmss").format(new java.util.Date())
    classifier.setTensorBoard(logDir = "/tmp/zoo/tcl", appName = config.encoder + "/" + date)
    classifier.compile(
      optimizer = new Adam(learningRate = config.learningRate),
      loss = SparseCategoricalCrossEntropy[Float](),
      metrics = List(new SparseCategoricalAccuracy[Float]())
    )

    val transformedValidationSet = validationSet.tokenize()
      .setWordIndex(wordIndex).word2idx()
      .shapeSequence(config.maxSequenceLength)
      .generateSample()

    classifier.fit(transformedTrainingSet, batchSize = config.batchSize, nbEpoch = config.epochs, transformedValidationSet)
    classifier.saveModel(config.modelPath + config.encoder + ".bin", overWrite = true)
    println("Finish training model and saving word dictionary.")
    classifier
  }

  def predict(textSet: TextSet, classifier: TextClassifier[Float]): TextSet = {
    val transformedTextSet = textSet.tokenize().loadWordIndex(config.modelPath + "/wordIndex.txt").word2idx()
      .shapeSequence(config.maxSequenceLength).generateSample()
    classifier.predict(transformedTextSet, batchPerThread = config.partitions)
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
    * Reads the vnExpress 5-category corpus (of 344,32 news articles) and
    * build a TextSet.
    *
    * @param sparkSession
    * @param path path to the data file(s)
    * @return a data frame of two columns (category, text)
    */
  def readJsonData(sparkSession: SparkSession, path: String): TextSet = {
        // each .json file is read to a df and these dfs are concatenated to form a big df
    val filenames = new File(path).list().filter(_.endsWith(".json"))
    val dfs = filenames.map(f => sparkSession.read.json(path + f))
    val input = dfs.reduce(_ union _)
    import sparkSession.implicits._
    val categories = input.select("category").map(row => row.getString(0)).distinct.collect().sorted
    val labels = categories.zipWithIndex.toMap
    numLabels = labels.size
    println(s"Found ${numLabels} classes")
    println(labels.mkString(", "))
    println("Creating text set. Please wait...")
    val textRDD = input.rdd.map(row => TextFeature(row.getString(1).toLowerCase(), labels(row.getString(0))))
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
      opt[Int]('n', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("maximum sequence length for a text")
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

      MKL.setNumThreads(4)

      val app = new Classifier(sparkSession.sparkContext, config)
      val textSet = readJsonData(sparkSession, config.dataPath).toDistributed(sparkContext, config.partitions)
      val Array(training, validation) = textSet.randomSplit(Array(config.trainingSplit, 1 - config.trainingSplit))
      val validationMethods = Array(new SparseCategoricalAccuracy[Float]())

      config.mode match {
        case "train" =>
          val classifier = app.train(training, validation)
          classifier.setEvaluateStatus()           
          val prediction = app.predict(training, classifier)
          val accuracy = classifier.evaluate(prediction.toDistributed().rdd.map(_.getSample), validationMethods, batchSize = Some(config.batchSize))
          println(accuracy.mkString(", "))
        case "eval" =>
          val classifier = TextClassifier.loadModel[Float](config.modelPath + config.encoder + ".bin")
          classifier.setEvaluateStatus()           
          val prediction = app.predict(validation, classifier)
          val accuracy = classifier.evaluate(prediction.toDistributed().rdd.map(_.getSample), validationMethods, batchSize = Some(config.batchSize))
          println(accuracy.mkString(", "))
        case "predict" =>
          val textSet = readJsonData(sparkSession, config.dataPath).toDistributed(sparkContext, config.partitions)
          val classifier = TextClassifier.loadModel[Float](config.modelPath + config.encoder + ".bin")
          val prediction = app.predict(textSet, classifier)
          prediction.toLocal().array.take(10).foreach(println)
      }
      sparkSession.stop()
      case None =>
    }
  }
}