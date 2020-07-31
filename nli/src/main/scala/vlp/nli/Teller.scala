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
import breeze.linalg.Tensor
import com.intel.analytics.bigdl.nn.SplitTable
import com.intel.analytics.bigdl.nn.Concat
import com.intel.analytics.bigdl.nn.SelectTable
import _root_.com.intel.analytics.bigdl.nn.Pack
import com.intel.analytics.bigdl.tensor.Storage
import com.intel.analytics.bigdl.nn.Recurrent
import com.intel.analytics.bigdl.nn.Select
import com.intel.analytics.bigdl.nn.Squeeze
import com.intel.analytics.bigdl.nn.Sigmoid

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
    prepocessor.write.overwrite().save(Paths.get(config.modelPath, config.language, config.modelType).toString)
    logger.info("Pre-processing pipeline saved.")
    val df = prepocessor.transform(input)
    // determine the vocab size and dictionary
    val vocabulary = prepocessor.stages.last.asInstanceOf[CountVectorizerModel].vocabulary
    val vocabSize = Math.min(config.numFeatures, vocabulary.size) + 1 // plus one (see [[SequenceVectorizer]])
    val dictionary: Map[String, Int] = vocabulary.zipWithIndex.toMap

    // prepare the input data frame for BigDL model
    val dlInputDF = if (config.modelType == "seq") {
      val sequenceVectorizer = new SequenceVectorizer(dictionary, config.maxSequenceLength).setInputCol("tokens").setOutputCol("features")
      sequenceVectorizer.transform(df.select("label", "tokens"))
    } else {
      val premiseSequenceVectorizer = new SequenceVectorizer(dictionary, config.maxSequenceLength).setInputCol("premise").setOutputCol("premiseIndexVector")
      val hypothesisSequenceVectorizer = new SequenceVectorizer(dictionary, config.maxSequenceLength).setInputCol("hypothesis").setOutputCol("hypothesisIndexVector") 
      val vectorStacker = new VectorStacker().setInputCols(Array("premiseIndexVector", "hypothesisIndexVector")).setOutputCol("features")
      val pipeline = new Pipeline().setStages(Array(premiseSequenceVectorizer, hypothesisSequenceVectorizer, vectorStacker))
      val ef = df.select("label", "premise", "hypothesis")
      pipeline.fit(ef).transform(ef)
    }

    // add 1 to the 'label' column to get the 'category' column for BigDL model to train
    val increase = udf((x: Double) => (x + 1), DoubleType)
    val trainingDF = dlInputDF.withColumn("category", increase(dlInputDF("label")))
    trainingDF.show()

    val dlModel = if (config.modelType == "seq") sequentialTransducer(vocabSize, config.maxSequenceLength) else parallelTransducer(vocabSize, config.maxSequenceLength)

    val trainSummary = TrainSummary(appName = config.encoder, logDir = Paths.get("/tmp/nli/summary/", config.language, config.modelType).toString())
    val validationSummary = ValidationSummary(appName = config.encoder, logDir = Paths.get("/tmp/nli/summary/", config.language, config.modelType).toString())
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
    dlModel.saveModule(Paths.get(config.modelPath, config.language, config.modelType, s"${config.encoder}.bigdl").toString(), 
      Paths.get(config.modelPath, config.language, config.modelType, s"${config.encoder}.bin").toString(), true)
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
    config.encoder match {
      case "cnn" => 
        model.add(Convolution1D(config.encoderOutputSize, config.kernelWidth, activation = "relu"))
        model.add(GlobalMaxPooling1D())
      case "gru" => model.add(GRU(config.encoderOutputSize))
      case _ => throw new IllegalArgumentException(s"Unsupported encoder for Teller: $config.encoder")
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
    config.encoder match {
      case "cnn" => 
        premiseLayers.add(TemporalConvolution(config.embeddingSize, config.encoderOutputSize, config.kernelWidth)).add(Sigmoid())
        premiseLayers.add(TemporalMaxPooling(config.kernelWidth))
        premiseLayers.add(Select(2, -1)) // can replace -1 with (maxSeqLen - config.kernelWidth)/config.kernelWidth + 1
        hypothesisLayers.add(TemporalConvolution(config.embeddingSize, config.encoderOutputSize, config.kernelWidth)).add(Sigmoid())
        hypothesisLayers.add(TemporalMaxPooling(config.kernelWidth))
        hypothesisLayers.add(Select(2, -1))
      case "gru" => 
        val pRecur = Recurrent().add(com.intel.analytics.bigdl.nn.GRU(config.embeddingSize, config.encoderOutputSize))
        premiseLayers.add(pRecur).add(Select(2, maxSeqLen))
        val hRecur = Recurrent().add(com.intel.analytics.bigdl.nn.GRU(config.embeddingSize, config.encoderOutputSize))
        hypothesisLayers.add(hRecur).add(Select(2, maxSeqLen))
    }
    branches.add(premiseLayers).add(hypothesisLayers)

    model.add(branches)
      .add(JoinTable(2, 2))
      .add(Linear(2*config.encoderOutputSize, config.numLabels))
      .add(SoftMax())
  }
}

object Teller {

  def test(): Unit = {
    val model = Sequential().add(Reshape(Array(2, 4))).add(SplitTable(1))
    val branches = ParallelTable()
    val u = Recurrent().add(com.intel.analytics.bigdl.nn.GRU(3, 5))
    val v = Recurrent().add(com.intel.analytics.bigdl.nn.GRU(3, 5))
    val first = Sequential().add(LookupTable(8, 3)).add(Reshape(Array(1,4,3))).add(u).add(Squeeze(1)).add(Select(1,4))
    val second = Sequential().add(LookupTable(8, 3)).add(Reshape(Array(1,4,3))).add(v).add(Squeeze(1)).add(Select(1,4))
    branches.add(first).add(second)
    model.add(branches)
      //.add(Pack(1))
      .add(JoinTable(1,1))
    val input = com.intel.analytics.bigdl.tensor.Tensor(Storage(Array(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)), 1, Array(8))
    val output = model.forward(input)
    println(output)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

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
      opt[String]('e', "encoder").action((x, conf) => conf.copy(encoder = x)).text("type of encoder, either cnn or gru")
      opt[Int]('o', "encoderOutputSize").action((x, conf) => conf.copy(encoderOutputSize = x)).text("output size of the encoder")
      opt[Int]('n', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("maximum sequence length for a text")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
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