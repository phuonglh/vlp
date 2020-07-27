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
import com.intel.analytics.bigdl.nn.keras.GRU
import java.nio.file.Paths
import com.intel.analytics.bigdl.optim.Adam
import org.apache.spark.ml.feature.StringIndexer
import com.intel.analytics.bigdl.optim.Trigger
import com.intel.analytics.bigdl.optim.Top1Accuracy
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.DoubleType
import com.intel.analytics.bigdl.visualization.TrainSummary
import com.intel.analytics.bigdl.visualization.ValidationSummary


class Teller(sparkSession: SparkSession, config: ConfigTeller) {
  final val logger = LoggerFactory.getLogger(getClass.getName)

  def train(input: DataFrame): Unit = {
    val labelIndexer = new StringIndexer().setInputCol("gold_label").setOutputCol("label")
    val premiseTokenizer = new Tokenizer().setInputCol("sentence1_tokenized").setOutputCol("premise")
    val hypothesisTokenizer = new Tokenizer().setInputCol("sentence2_tokenized").setOutputCol("hypothesis")
    val sequenceAssembler = new SequenceAssembler().setInputCols(Array("premise", "hypothesis")).setOutputCol("xs")
    val countVectorizer = new CountVectorizer().setInputCol("xs").setOutputCol("features")
      .setVocabSize(config.numFeatures)
      .setMinDF(config.minFrequency)
      .setBinary(true)

    val prePipeline = new Pipeline().setStages(Array(labelIndexer, premiseTokenizer, hypothesisTokenizer, sequenceAssembler, countVectorizer))
    logger.info("Fitting pre-processing pipeline...")
    val prepocessor = prePipeline.fit(input)
    prepocessor.write.overwrite().save(Paths.get(config.modelPath, config.modelType).toString)
    logger.info("Pre-processing pipeline saved.")
    val df = prepocessor.transform(input)

    // add 1 to the 'label' column to get the 'category' column for BigDL model to train
    val increase = udf((x: Double) => (x + 1), DoubleType)
    val trainingDF = df.withColumn("category", increase(df("label")))
    trainingDF.show(false)

    val dlModel = sequentialTransducer()
    val trainSummary = TrainSummary(appName = "teller", logDir = "/tmp/nli/summary/")
    val validationSummary = ValidationSummary(appName = "teller", logDir = "/tmp/nli/summary/")
    val classifier = new DLClassifier(dlModel, ClassNLLCriterion[Float](), Array(config.numFeatures))
      .setLabelCol("category")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(config.learningRate))
      .setTrainSummary(trainSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, trainingDF, Array(new Top1Accuracy), config.batchSize)
  
    val model = classifier.fit(trainingDF)
    dlModel.saveModule(Paths.get(config.modelPath, "nli.bigdl").toString(), Paths.get(config.modelPath, "nli.bin").toString(), true)
  }

  def sequentialTransducer(): Module[Float] = {
    val model = Sequential()
    val embedding = Embedding(config.numFeatures, config.embeddingSize, inputShape = Shape(config.numFeatures))
    model.add(embedding)
      .add(GRU(config.outputSize))
      .add(Dense(config.numLabels, activation = "relu"))
  }

  def parallelTransducer(): Module[Float] = {
    val model = Sequential()
    val branches = ParallelTable()
    val sourceEmbedding = Embedding(config.numFeatures, config.embeddingSize, inputShape = Shape(config.numFeatures))
    val source = Sequential().add(sourceEmbedding).add(GRU(config.outputSize))
    val targetEmbedding = Embedding(config.numFeatures, config.embeddingSize, inputShape = Shape(config.numFeatures))
    val target = Sequential().add(GRU(config.outputSize))
    branches.add(source).add(target)
    model.add(branches).add(ConcatTable()).add(Dense(config.numLabels, activation = "softmax"))
  }

}

object Teller {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val config = ConfigTeller()
    val conf = Engine.createSparkConf().setAppName("vlp.nli.Teller").setMaster(config.master)
    val sc = new SparkContext(conf)
    Engine.init
    val sparkSession = SparkSession.builder().getOrCreate()
    val df = sparkSession.read.json(config.dataPath).select("gold_label", "sentence1_tokenized", "sentence2_tokenized")
    df.show(false)
    val teller = new Teller(sparkSession, config)
    teller.train(df)
    sparkSession.stop()
  }
}