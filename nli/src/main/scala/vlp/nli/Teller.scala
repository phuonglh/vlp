package vlp.nli

import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.utils.Shape
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.functions.udf
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import java.nio.file.Paths
import org.apache.spark.ml.feature.StringIndexer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.CountVectorizerModel

import org.slf4j.LoggerFactory

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.keras.{GRU, Embedding, Dense, Convolution1D, GlobalMaxPooling1D}
import com.intel.analytics.bigdl.nn.keras.{Sequential => SequentialKeras, Reshape => ReshapeKeras}
import com.intel.analytics.bigdl.nn.{Sequential, Reshape, Transpose}
import com.intel.analytics.bigdl.nn.Transpose
import com.intel.analytics.bigdl.nn.Linear
import com.intel.analytics.bigdl.nn.LookupTable
import com.intel.analytics.bigdl.nn.TemporalConvolution
import com.intel.analytics.bigdl.nn.TemporalMaxPooling
import com.intel.analytics.bigdl.nn.JoinTable
import com.intel.analytics.bigdl.nn.ReLU
import com.intel.analytics.bigdl.nn.SoftMax
import com.intel.analytics.bigdl.nn.ParallelTable
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dlframes.DLClassifier
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.optim.Trigger
import com.intel.analytics.bigdl.optim.Top1Accuracy
import com.intel.analytics.bigdl.visualization.TrainSummary
import com.intel.analytics.bigdl.visualization.ValidationSummary

import scopt.OptionParser
import com.intel.analytics.bigdl.nn.Echo
import com.intel.analytics.bigdl.nn.SplitTable
import com.intel.analytics.bigdl.nn.Concat
import com.intel.analytics.bigdl.nn.SelectTable
import _root_.com.intel.analytics.bigdl.nn.Pack
import com.intel.analytics.bigdl.tensor.Storage
import com.intel.analytics.bigdl.nn.Recurrent
import com.intel.analytics.bigdl.nn.Select
import com.intel.analytics.bigdl.nn.Squeeze
import com.intel.analytics.bigdl.nn.SpatialConvolution
import com.intel.analytics.bigdl.nn.SpatialMaxPooling
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.json4s._
import org.json4s.jackson.Serialization
import java.nio.file.Files
import java.nio.file.StandardOpenOption


/**
  * Natural Language Inference module
  * phuonglh, July 2020.
  * <phuonglh@gmail.com>
  *
  * @param sparkSession
  * @param config
  */

class Teller(sparkSession: SparkSession, config: ConfigTeller) {
  final val logger = LoggerFactory.getLogger(getClass.getName)

  def train(training: DataFrame, test: DataFrame): Scores = {
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
    val prepocessor = prePipeline.fit(training)
    prepocessor.write.overwrite().save(Paths.get(config.modelPath, config.language, config.modelType).toString)
    logger.info("Pre-processing pipeline saved.")
    // determine the vocab size and dictionary
    val vocabulary = prepocessor.stages.last.asInstanceOf[CountVectorizerModel].vocabulary
    val vocabSize = Math.min(config.numFeatures, vocabulary.size) + 1 // plus one (see [[SequenceVectorizer]])
    val dictionary: Map[String, Int] = vocabulary.zipWithIndex.toMap

    // prepare the training and test data frames for BigDL model
    val df = prepocessor.transform(training)
    val tdf = prepocessor.transform(test)
    val (dlTrainingDF, dlTestDF) = if (config.modelType == "seq") {
      val sequenceVectorizer = new SequenceVectorizer(dictionary, config.maxSequenceLength).setInputCol("tokens").setOutputCol("features")
      (sequenceVectorizer.transform(df.select("label", "tokens")), sequenceVectorizer.transform(tdf.select("label", "tokens")))
    } else {
      val premiseSequenceVectorizer = new SequenceVectorizer(dictionary, config.maxSequenceLength).setInputCol("premise").setOutputCol("premiseIndexVector")
      val hypothesisSequenceVectorizer = new SequenceVectorizer(dictionary, config.maxSequenceLength).setInputCol("hypothesis").setOutputCol("hypothesisIndexVector") 
      val vectorStacker = new VectorStacker().setInputCols(Array("premiseIndexVector", "hypothesisIndexVector")).setOutputCol("features")
      val pipeline = new Pipeline().setStages(Array(premiseSequenceVectorizer, hypothesisSequenceVectorizer, vectorStacker))
      val ef = df.select("label", "premise", "hypothesis")
      val tef = tdf.select("label", "premise", "hypothesis")
      val pm = pipeline.fit(ef)
      (pm.transform(ef), pm.transform(tef))
    }

    // add 1 to the 'label' column to get the 'category' column for BigDL model to train
    val increase = udf((x: Double) => (x + 1), DoubleType)
    val trainingDF = dlTrainingDF.withColumn("category", increase(dlTrainingDF("label")))
    val testDF = dlTestDF.withColumn("category", increase(dlTestDF("label")))
    trainingDF.show()
    testDF.show()

    val dlModel = if (config.modelType == "seq") sequentialTransducer(vocabSize, config.maxSequenceLength) else parallelTransducer(vocabSize, config.maxSequenceLength)

    val trainSummary = TrainSummary(appName = config.encoderType, logDir = Paths.get("/tmp/nli/summary/", config.language, config.modelType).toString())
    val validationSummary = ValidationSummary(appName = config.encoderType, logDir = Paths.get("/tmp/nli/summary/", config.language, config.modelType).toString())
    val featureSize = if (config.modelType == "seq") Array(config.maxSequenceLength) else Array(2*config.maxSequenceLength)
    val classifier = new DLClassifier(dlModel, ClassNLLCriterion[Float](), featureSize)
      .setLabelCol("category")
      .setFeaturesCol("features")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(config.learningRate))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, trainingDF, Array(new Top1Accuracy), config.batchSize)
  
    val model = classifier.fit(trainingDF)
    dlModel.saveModule(Paths.get(config.modelPath, config.language, config.modelType, s"${config.encoderType}.bigdl").toString(), 
      Paths.get(config.modelPath, config.language, config.modelType, s"${config.encoderType}.bin").toString(), true)

    val prediction = model.transform(testDF)
    prediction.show()
    import sparkSession.implicits._
    val predictionAndLabels = prediction.select("category", "prediction").map(row => (row.getDouble(0), row.getDouble(1))).rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val scores = (metrics.accuracy, metrics.weightedFMeasure)
    logger.info(s"scores = $scores")
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
    val xs = validationSummary.readScalar("Top1Accuracy").map(_._2)
    Scores(arch = config.modelType, encoder = config.encoderType, maxSequenceLength = config.maxSequenceLength, embeddingSize = config.embeddingSize, 
      encoderSize = config.encoderOutputSize, trainingScores = xs, testScore = scores._2)
  }

  /**
    * Constructs a sequential model for NLI using Keras-style layers.
    *
    * @param vocabSize
    * @param maxSeqLen
    * @return a BigDL Keras-style model
    */
  def sequentialTransducer(vocabSize: Int, maxSeqLen: Int): Module[Float] = {
    val model = SequentialKeras()
    val embedding = Embedding(vocabSize, config.embeddingSize, inputShape = Shape(maxSeqLen))
    model.add(embedding)
    config.encoderType match {
      case "cnn" => 
        model.add(Convolution1D(config.encoderOutputSize, config.filterSize, activation = "relu"))
        model.add(GlobalMaxPooling1D())
      case "gru" => model.add(GRU(config.encoderOutputSize))
      case _ => throw new IllegalArgumentException(s"Unsupported encoder type for Teller: $config.encoderType")
    }
    model.add(Dense(config.numLabels, activation = "softmax"))
  }

  /**
    * Constructs a parallel model for NLI using core BigDL layers.
    *
    * @param vocabSize
    * @param maxSeqLen
    * @return a BigDL model
    */
  def parallelTransducer(vocabSize: Int, maxSeqLen: Int): Module[Float] = {
    val model = new Sequential().add(Reshape(Array(2, maxSeqLen))).add(SplitTable(2, 3))     
    val branches = ParallelTable()
    val premiseLayers = Sequential().add(LookupTable(vocabSize, config.embeddingSize))
    val hypothesisLayers = Sequential().add(LookupTable(vocabSize, config.embeddingSize))
    config.encoderType match {
      case "cnn" => 
        premiseLayers.add(Reshape(Array(maxSeqLen, 1, config.embeddingSize)))
        premiseLayers.add(SpatialConvolution(config.embeddingSize, config.encoderOutputSize, kernelW = 1, kernelH = config.filterSize, 1, 1, -1, -1, format = DataFormat.NHWC)) 
        premiseLayers.add(Squeeze(3))
        premiseLayers.add(ReLU())
        premiseLayers.add(Reshape(Array(maxSeqLen, 1, config.encoderOutputSize)))
        premiseLayers.add(SpatialMaxPooling(1, maxSeqLen, format = DataFormat.NHWC))
        premiseLayers.add(Squeeze(3))
        premiseLayers.add(Squeeze(2))
        hypothesisLayers.add(Reshape(Array(maxSeqLen, 1, config.embeddingSize)))
        hypothesisLayers.add(SpatialConvolution(config.embeddingSize, config.encoderOutputSize, kernelW = 1, kernelH = config.filterSize, 1, 1, -1, -1, format = DataFormat.NHWC)) 
        hypothesisLayers.add(Squeeze(3))
        hypothesisLayers.add(ReLU())
        hypothesisLayers.add(Reshape(Array(maxSeqLen, 1, config.encoderOutputSize)))
        hypothesisLayers.add(SpatialMaxPooling(1, maxSeqLen, format = DataFormat.NHWC))
        hypothesisLayers.add(Squeeze(3))
        hypothesisLayers.add(Squeeze(2))
      case "gru" => 
        val pRecur = Recurrent().add(com.intel.analytics.bigdl.nn.GRU(config.embeddingSize, config.encoderOutputSize))
        premiseLayers.add(pRecur).add(Select(2, -1))
        val hRecur = Recurrent().add(com.intel.analytics.bigdl.nn.GRU(config.embeddingSize, config.encoderOutputSize))
        hypothesisLayers.add(hRecur).add(Select(2, -1))
    }
    branches.add(premiseLayers).add(hypothesisLayers)

    model.add(branches)
      .add(JoinTable(2, 2))
      .add(Linear(2*config.encoderOutputSize, config.numLabels))
      .add(SoftMax())
  }
}

object Teller {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    implicit val formats = Serialization.formats(NoTypeHints)

    val parser = new OptionParser[ConfigTeller]("vlp.nli.Teller") {
      head("vlp.nli.Teller", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/predict")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("data path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/zoo/tcl/'")
      opt[Int]('w', "embeddingSize").action((x, conf) => conf.copy(embeddingSize = x)).text("embedding size")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type, either seq or par")
      opt[String]('e', "encoderType").action((x, conf) => conf.copy(encoderType = x)).text("type of encoder, either 'cnn' or 'gru'")
      opt[Int]('o', "encoderOutputSize").action((x, conf) => conf.copy(encoderOutputSize = x)).text("output size of the encoder")
      opt[Int]('n', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("maximum sequence length for a text")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Unit]('v', "verbose").action((x, conf) => conf.copy(verbose = true)).text("verbose mode")
    }
    parser.parse(args, ConfigTeller()) match {
      case Some(config) =>
        val sparkConfig = Engine.createSparkConf()
          .setMaster(config.master)
          .set("spark.executor.memory", config.executorMemory)
          .setAppName("nli.Teller")
        val sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate()
        val sparkContext = sparkSession.sparkContext
        Engine.init
        val df = sparkSession.read.json(config.dataPath).select("gold_label", "sentence1_tokenized", "sentence2_tokenized")
        df.groupBy("gold_label").count().show(false)
        val Array(training, test) = df.randomSplit(Array(0.8, 0.2), seed = 12345)
        val teller = new Teller(sparkSession, config)
        config.mode match {
          case "train" => 
            teller.train(training, test)
          case "eval" => 
          case "predict" => 
          case "experiments" => 
            val maxSequenceLengths = Array(40, 50, 60)
            val embeddingSizes = Array(25, 50, 80, 100, 128, 150)
            val encoderOutputSizes = Array(25, 50, 80, 100, 128, 150, 200, 256, 300, 400, 500)
            for (n <- maxSequenceLengths)
              for (d <- embeddingSizes)
                for (o <- encoderOutputSizes) {
                  val conf = ConfigTeller(modelType = config.modelType, encoderType = config.encoderType, maxSequenceLength = n, embeddingSize = d, encoderOutputSize = o)
                  val teller = new Teller(sparkSession, conf)
                  val scores = teller.train(training, test)
                  val content = Serialization.writePretty(scores) + ",\n"
                  Files.write(Paths.get("dat/nli/scores.nli.json"), content.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
                }
          case _ => System.err.println("Unsupported mode!")
        }
        sparkSession.stop()
      case None => 
    }
  }
}

case class Scores(
  arch: String,
  encoder: String,
  maxSequenceLength: Int,
  embeddingSize: Int,
  encoderSize: Int,
  trainingScores: Array[Float],
  testScore: Double
)