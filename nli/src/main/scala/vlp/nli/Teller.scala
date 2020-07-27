package vlp.nli

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.Module
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.slf4j.LoggerFactory
import com.intel.analytics.bigdl.nn.keras.{Sequential, Embedding, Dense, Reshape}
import com.intel.analytics.bigdl.nn.ParallelTable
import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.nn.Linear
import com.intel.analytics.bigdl.nn.ReLU
import com.intel.analytics.bigdl.nn.ConcatTable
import com.intel.analytics.bigdl.nn.SoftMax
import com.intel.analytics.bigdl.utils.Shape
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.CountVectorizer
import com.intel.analytics.bigdl.dlframes.DLClassifier
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import org.apache.spark.sql.functions.udf
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.nn.LogSoftMax
import com.intel.analytics.bigdl.nn.keras.{GRU, LSTM, Convolution1D, GlobalMaxPooling1D}
import java.nio.file.Paths
import com.intel.analytics.bigdl.optim.Adam
import org.apache.spark.ml.feature.StringIndexer
import com.intel.analytics.bigdl.optim.Trigger
import com.intel.analytics.bigdl.optim.Top1Accuracy
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.DoubleType
import com.intel.analytics.bigdl.visualization.TrainSummary
import com.intel.analytics.bigdl.visualization.ValidationSummary
import scopt.OptionParser
import org.apache.spark.ml.feature.CountVectorizerModel
import com.intel.analytics.bigdl.nn.keras.Dropout
import com.intel.analytics.bigdl.nn.keras.Activation

class Teller(sparkSession: SparkSession, config: ConfigTeller) {
  final val logger = LoggerFactory.getLogger(getClass.getName)

  def train(input: DataFrame): Unit = {
    val labelIndexer = new StringIndexer().setInputCol("gold_label").setOutputCol("label")
    val premiseTokenizer = new Tokenizer().setInputCol("sentence1_tokenized").setOutputCol("premise")
    val hypothesisTokenizer = new Tokenizer().setInputCol("sentence2_tokenized").setOutputCol("hypothesis")
    val sequenceAssembler = new SequenceAssembler().setInputCols(Array("premise", "hypothesis")).setOutputCol("tokens")
    val countVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("countVector")
      .setVocabSize(config.numFeatures)
      .setMinDF(config.minFrequency)
      .setBinary(true)

    val prePipeline = new Pipeline().setStages(Array(labelIndexer, premiseTokenizer, hypothesisTokenizer, sequenceAssembler, countVectorizer))
    logger.info("Fitting pre-processing pipeline...")
    val prepocessor = prePipeline.fit(input)
    prepocessor.write.overwrite().save(Paths.get(config.modelPath, config.modelType).toString)
    logger.info("Pre-processing pipeline saved.")
    val df = prepocessor.transform(input)

    // determine the vocab size
    val vocabulary = prepocessor.stages.last.asInstanceOf[CountVectorizerModel].vocabulary
    val vocabSize = Math.min(config.numFeatures, vocabulary.size) + 1 // plus one (see [[SequenceVectorizer]])

    // create a dictionary 
    val dictionary: Map[String, Int] = vocabulary.zipWithIndex.toMap
    val sequenceVectorizer = new SequenceVectorizer(dictionary, config.maxSequenceLength).setInputCol("tokens").setOutputCol("indexVector")
    // create input data frame for the BigDL model
    val dlInputDF = sequenceVectorizer.transform(df.select("label", "tokens"))

    // add 1 to the 'label' column to get the 'category' column for BigDL model to train
    val increase = udf((x: Double) => (x + 1), DoubleType)
    val trainingDF = dlInputDF.withColumn("category", increase(dlInputDF("label")))
    trainingDF.show()

    val dlModel = sequentialTransducer(vocabSize, config.maxSequenceLength)
    val trainSummary = TrainSummary(appName = config.encoder, logDir = "/tmp/nli/summary/" + config.language)
    val validationSummary = ValidationSummary(appName = config.encoder, logDir = "/tmp/nli/summary/" + config.language)
    val classifier = new DLClassifier(dlModel, ClassNLLCriterion[Float](), Array(config.maxSequenceLength))
      .setLabelCol("category")
      .setFeaturesCol("indexVector")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(config.learningRate))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, trainingDF, Array(new Top1Accuracy), config.batchSize)
  
    val model = classifier.fit(trainingDF)
    dlModel.saveModule(Paths.get(config.modelPath, config.language, config.modelType, "nli.bigdl").toString(), 
      Paths.get(config.modelPath, config.language, config.modelType, "nli.bin").toString(), true)
  }

  def sequentialTransducer(vocabSize: Int, featureSize: Int): Module[Float] = {
    val model = Sequential()
    val embedding = Embedding(vocabSize, config.embeddingSize, inputShape = Shape(featureSize))
    model.add(embedding)
    config.encoder match {
      case "cnn" => 
        model.add(Convolution1D(config.encoderOutputSize, 5, activation = "relu"))
        model.add(GlobalMaxPooling1D())
      case "gru" => model.add(GRU(config.encoderOutputSize))
      case "lstm" => model.add(LSTM(config.encoderOutputSize))
      case _ => throw new IllegalArgumentException(s"Unsupported encoder for Teller: $config.encoder")
    }
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(Dense(config.numLabels, activation = "softmax"))
  }

  def parallelTransducer(vocabSize: Int, featureSize: Int): Module[Float] = {
    val model = Sequential()
    val branches = ParallelTable()
    val sourceEmbedding = Embedding(vocabSize, config.embeddingSize, inputShape = Shape(featureSize))
    val source = Sequential().add(sourceEmbedding).add(GRU(config.encoderOutputSize))
    val targetEmbedding = Embedding(featureSize, config.embeddingSize, inputShape = Shape(featureSize))
    val target = Sequential().add(GRU(config.encoderOutputSize))
    branches.add(source).add(target)
    model.add(branches).add(ConcatTable()).add(Dense(config.numLabels, activation = "softmax"))
  }

}

object Teller {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val parser = new OptionParser[ConfigTeller]("vlp.nli.Teller") {
      head("vlp.nli.Teller", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/predict")
      opt[String]('e', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("data path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/zoo/tcl/'")
      opt[Int]('w', "embeddingSize").action((x, conf) => conf.copy(embeddingSize = x)).text("embedding size")
      opt[String]('t', "encoder").action((x, conf) => conf.copy(encoder = x)).text("type of encoder, either cnn, lstm or gru")
      opt[Int]('o', "encoderOutputSize").action((x, conf) => conf.copy(encoderOutputSize = x)).text("output size of the encoder")
      opt[Int]('n', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("maximum sequence length for a text")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
    }
    parser.parse(args, ConfigTeller()) match {
      case Some(config) =>
        val sparkConfig = Engine.createSparkConf()
          .setMaster(config.master)
          .set("spark.executor.memory", config.executorMemory)
          .setAppName("SHINRA")
        val sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate()
        val sparkContext = sparkSession.sparkContext
        Engine.init
        val df = sparkSession.read.json(config.dataPath).select("gold_label", "sentence1_tokenized", "sentence2_tokenized")
        df.groupBy("gold_label").count().show(false)
        val teller = new Teller(sparkSession, config)
        config.mode match {
          case "train" => 
            teller.train(df)
          case "eval" => 
          case "predict" => 
          case _ => 
        }
        sparkSession.stop()
      case None => 
    }
  }
}