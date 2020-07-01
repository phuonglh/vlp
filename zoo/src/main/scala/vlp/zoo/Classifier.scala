package vlp.zoo

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}
import org.slf4j.LoggerFactory

import com.intel.analytics.bigdl.optim.{Adagrad, Top1Accuracy}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.feature.text.{TextFeature, TextSet}
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.zoo.pipeline.api.keras.metrics.{Accuracy, Top5Accuracy}
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import scopt.OptionParser
import com.intel.analytics.bigdl.optim.Adam

/**
  * Configuration parameters of a Neural Network Classifier.
  *
  * @param master Spark master
  * @param mode train/eval/predict
  * @param dataPath path to the corpus (20news-18828)
  * @param gloveEmbeddingPath path to the GloVe pre-trained word embeddings
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
  dataPath: String = "/opt/data/news20/20news-18828/",
  gloveEmbeddingPath: String = "/opt/data/emb/glove.6B.200d.txt",
  modelPath: String = "dat/zoo/tcl/", // need the last back slash
  numFeatures: Int = 8192,
  encoder: String = "cnn",
  encoderOutputDimension: Int = 256,
  maxSequenceLength: Int = 500,
  trainingSplit: Double = 0.8,
  batchSize: Int = 64,
  epochs: Int = 20,
  learningRate: Double = 0.001,
  partitions: Int = 1,
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
    * Trains a neural text classifier on a text set. Trained model and word index are saved to
    * an external path specified in the configuration.
    * @param textSet
    */
  def train(textSet: TextSet): Unit = {
    println("Processing text data set...")
    val transformedTextSet = textSet.tokenize().normalize()
      .word2idx(10, maxWordsNum = config.numFeatures, minFreq = config.minFrequency)
      .shapeSequence(config.maxSequenceLength).generateSample()
    val wordIndex = transformedTextSet.getWordIndex

    val numLabels = textSet.toLocal().array.map(textFeature => textFeature.getLabel).toSet.size

    val classifier = TextClassifier(numLabels, config.gloveEmbeddingPath, wordIndex, config.maxSequenceLength, config.encoder, config.encoderOutputDimension)
    classifier.setTensorBoard(logDir = "/tmp/zoo/", appName = "zoo.tcl")
    classifier.compile(
      optimizer = new Adam(learningRate = config.learningRate),
      loss = SparseCategoricalCrossEntropy[Float](),
      metrics = List(new Accuracy())
    )
    val Array(training, validation) = transformedTextSet.randomSplit(Array(config.trainingSplit, 1 - config.trainingSplit))
    classifier.fit(training, batchSize = config.batchSize, nbEpoch = config.epochs, validation)
    classifier.saveModel(config.modelPath + config.encoder + ".bin", overWrite = true)
    transformedTextSet.saveWordIndex(config.modelPath + "/wordIndex.txt")
    println("Trained model and word dictionary saved.")
  }

  def predict(textSet: TextSet, classifier: TextClassifier[Float]): TextSet = {
    val transformedTextSet = textSet.tokenize().normalize().loadWordIndex(config.modelPath + "/wordIndex.txt").word2idx()
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

  def main(args: Array[String]): Unit = {
    val sparkConfig = Engine.createSparkConf()
      .setMaster("local[*]")
      .setAppName("Neural Text Classifier")
      // .set("spark.driver.host", "localhost")
    val sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate()
    val sparkContext = sparkSession.sparkContext
    Engine.init

    val parser = new OptionParser[ConfigClassifier]("zoo.tcl") {
      head("vlp.zoo.Classifier", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/predict")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("data path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/zoo/tcl/'")
      opt[String]('e', "encoder").action((x, conf) => conf.copy(encoder = x)).text("encoder, either cnn, lstm or gru")
      opt[Int]('o', "encoderOutputDimension").action((x, conf) => conf.copy(encoderOutputDimension = x)).text("output dimension of the encoder")
      opt[Int]('n', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("maximum sequence length for a text")
    }
    parser.parse(args, ConfigClassifier()) match {
      case Some(config) =>
        val app = new Classifier(sparkSession.sparkContext, config)
        config.mode match {
          case "train" =>
            val textSet = TextSet.read(config.dataPath).toDistributed(sparkContext, config.partitions)
            app.train(textSet)
          case "eval" =>
            val textSet = TextSet.read(config.dataPath).toDistributed(sparkContext, config.partitions)
              .loadWordIndex(config.modelPath + "/wordIndex.txt")
            val classifier = TextClassifier.loadModel[Float](config.modelPath + config.encoder + ".bin")
            classifier.setEvaluateStatus()           
            val validationMethods = Array(new Top1Accuracy[Float](), new Top5Accuracy[Float]())
            val prediction = app.predict(textSet, classifier)
            val accuracy = classifier.evaluate(prediction.toDistributed().rdd.map(_.getSample), validationMethods)
            println(accuracy.mkString(", "))
          case "predict" =>
            val textSet = TextSet.read(config.dataPath).toDistributed(sparkContext, config.partitions)
              .loadWordIndex(config.modelPath + "/wordIndex.txt")
            val classifier = TextClassifier.loadModel[Float](config.modelPath + config.encoder + ".bin")
            val prediction = app.predict(textSet, classifier)
            prediction.toLocal().array.take(10).foreach(println)
        }
      case None =>
    }
    sparkSession.stop()
  }

}