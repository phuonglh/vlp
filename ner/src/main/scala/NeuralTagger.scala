
package vlp.ner

import org.apache.spark.sql.SparkSession

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn.keras.{GRU, Embedding, Dense, SoftMax}
import com.intel.analytics.bigdl.nn.keras.Sequential
import com.intel.analytics.bigdl.nn.keras.TimeDistributed
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.Shape

import org.slf4j.LoggerFactory

import scopt.OptionParser
import org.apache.log4j.Level
import org.apache.log4j.Logger

import org.json4s._
import org.json4s.jackson.Serialization
import com.intel.analytics.bigdl.nn.TimeDistributedCriterion
import com.intel.analytics.bigdl.dlframes.DLEstimator
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.optim.Trigger
import com.intel.analytics.bigdl.optim.Top1Accuracy
import org.apache.spark.ml.linalg.Vectors
import com.intel.analytics.bigdl.optim.ValidationMethod
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.ValidationResult
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.optim.AccuracyResult

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.CountVectorizerModel
import java.nio.file.Paths
import org.apache.spark.ml.PipelineModel
import com.intel.analytics.bigdl.dlframes.DLModel
import java.nio.file.Files
import java.nio.file.StandardOpenOption

/**
  * A neural named entity tagger for Vietnamese.
  * 
  * phuonglh
  * 
  */
class NeuralTagger(sparkSession: SparkSession, config: ConfigNER) {
  val logger = LoggerFactory.getLogger(getClass.getName)
  val prefix = Paths.get(config.modelPath, config.language, "gru", s"${config.outputSize}").toString()

  import sparkSession.implicits._

  def createDataFrame(dataPath: String): DataFrame = {
    val sentences = CorpusReader.readCoNLL(dataPath)
    val seqPairs = sentences.map(sentence => {
      val tokens = sentence.tokens
      (tokens.map(_.word).mkString(" "), tokens.map(_.namedEntity).mkString(" "))
    })
    sparkSession.createDataFrame(seqPairs).toDF("x", "y")
  }

  def train(training: DataFrame, test: DataFrame): Module[Float] = {
    // build a preprocessing pipeline and determine the dictionary and vocab size
    val wordTokenizer = new Tokenizer().setInputCol("x").setOutputCol("words")
    val wordCountVectorizer = new CountVectorizer().setInputCol("words").setOutputCol("countVector").setMinDF(config.minFrequency)
    val labelTokenizer = new Tokenizer().setInputCol("y").setOutputCol("labels")
    val labelCountVectorizer = new CountVectorizer().setInputCol("labels").setOutputCol("labelVector").setMinDF(config.minFrequency)
    val preprocessingPipeline = new Pipeline().setStages(Array(wordTokenizer, wordCountVectorizer, labelTokenizer, labelCountVectorizer))
    val preprocessingPipelineModel = preprocessingPipeline.fit(training)  
    val (trainingAlpha, testAlpha) = (preprocessingPipelineModel.transform(training), preprocessingPipelineModel.transform(test))
    if (config.verbose) {
      trainingAlpha.show()
      testAlpha.show()
    }
    preprocessingPipelineModel.write.overwrite.save(Paths.get(config.modelPath, config.language, "gru").toString())

    val wordDictionary = preprocessingPipelineModel.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex.toMap
    val vocabSize = wordDictionary.size
    val labelDictionary = preprocessingPipelineModel.stages(3).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex.toMap
    val labelSize = labelDictionary.size
    logger.info(labelDictionary.toString)
    logger.info(s"vocabSize = ${vocabSize}")

    // transform sequences of words/labels into vectors of indices for use in the DL model
    val wordSequenceVectorizer = new SequenceVectorizer(wordDictionary, config.maxSequenceLength).setInputCol("words").setOutputCol("features")
    val labelSequenceVectorizer = new SequenceVectorizer(labelDictionary, config.maxSequenceLength).setInputCol("labels").setOutputCol("label")
    val pipeline = new Pipeline().setStages(Array(wordSequenceVectorizer, labelSequenceVectorizer))
    val pipelineModel = pipeline.fit(trainingAlpha)
    val (trainingBeta, testBeta) = (pipelineModel.transform(trainingAlpha), pipelineModel.transform(testAlpha))
    if (config.verbose) {
      trainingBeta.show()
      testBeta.show()
    }
    // train a DL model
    val model = buildModel(vocabSize + 1, labelSize, config.maxSequenceLength)
    val trainSummary = TrainSummary(appName = "gru", logDir = "/tmp/ner/" + config.language)
    val validationSummary = ValidationSummary(appName = "gru", logDir = "/tmp/ner/" + config.language)
    val classifier = new DLEstimator(model, TimeDistributedCriterion(ClassNLLCriterion[Float]()), featureSize = Array(config.maxSequenceLength), labelSize = Array(config.maxSequenceLength))
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(0.001))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, testBeta, Array(new TimeDistributedTop1Accuracy()), config.batchSize)
    classifier.fit(trainingBeta)
    model.saveModule(prefix + ".bigdl", prefix + ".bin", true)
  }

  /**
    * Constructs a sequential model for NLI using Keras-style layers.
    *
    * @param vocabSize
    * @param labelSize
    * @param maxSeqLen
    * @return a BigDL Keras-style model
    */
  def buildModel(vocabSize: Int, labelSize: Int, maxSeqLen: Int): Module[Float] = {
    val model = Sequential()
    val embedding = Embedding(vocabSize, config.embeddingSize, inputShape = Shape(maxSeqLen))
    model.add(embedding)
    model.add(GRU(config.outputSize, returnSequences = true))
    model.add(TimeDistributed(Dense(labelSize, activation = "softmax")))
  }

  /**
   * Predicts label sequence given word sequence. The input data frame has 'x' column.
  */
  def predict(input: DataFrame, preprocessor: PipelineModel, model: DLModel[Float]): DataFrame = {
    val wordTokenizer = new Tokenizer().setInputCol("x").setOutputCol("words")
    val alpha = wordTokenizer.transform(input)
    val wordDictionary = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex.toMap
    val wordSequenceVectorizer = new SequenceVectorizer(wordDictionary, config.maxSequenceLength).setInputCol("words").setOutputCol("features")
    val beta = wordSequenceVectorizer.transform(alpha)
    val gamma = model.transform(beta)
    val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    val labelMap = (0 until labels.size).zip(labels).toMap
    val predictor = new Predictor(labelMap, config.maxSequenceLength).setInputCol("prediction").setOutputCol("z")
    predictor.transform(gamma)
  }

  def predict(inputPathCoNLL: String, preprocessor: PipelineModel, model: DLModel[Float]): Unit = {
    val inputDF = createDataFrame(inputPathCoNLL)
    val outputDF = predict(inputDF, preprocessor, model)
    val result = outputDF.select("words", "y", "z").map(row => {
      val n = row.getAs[Seq[String]](0).size
      val ys = row.getAs[String](1).split(" ")
      val zs = row.getAs[Seq[String]](2).take(n)
      val st = ys.zip(zs).map{case (y, z) => y + " " + z.toUpperCase()}.mkString("\n")
      st + "\n"
    }).collect()
    import scala.collection.JavaConversions._
    Files.write(Paths.get(config.output), result.toList, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }
}

/**
  * Time-distributed top 1 accuracy.
  *
  * @param paddingValue padding value for label sequence
  * @param ev
  */
class TimeDistributedTop1Accuracy(paddingValue: Int = -1)(implicit ev: TensorNumeric[Float]) extends ValidationMethod[Float] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    var correct = 0
    var count = 0
    val _output = output.asInstanceOf[Tensor[Float]]
    val _target = target.asInstanceOf[Tensor[Float]]
    _output.split(1).zip(_target.split(1)).foreach { case (tensor, ys) => 
      val zs = tensor.split(1).map { t =>
        val values = t.toArray()
        val k = (0 until values.size).zip(values).maxBy(p => p._2)._1
        k + 1 // one-based label index
      }
      // filter the padded value in the gold target before comparing with the prediction
      val c = ys.toArray().filter(e => e != paddingValue).zip(zs)
        .map(p => if (p._1 == p._2) 1 else 0)
      correct += c.sum
      count += c.size
    }
    new AccuracyResult(correct, count)
  }
  override def format(): String = "TimeDistributedTop1Accuracy"
}

object NeuralTagger {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    implicit val formats = Serialization.formats(NoTypeHints)

    val parser = new OptionParser[ConfigNER]("vlp.ner.NeuralTagger") {
      head("vlp.ner.NeuralTagger", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/predict")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language, either 'vie' or 'eng'")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("data path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/zoo/tcl/'")
      opt[String]('e', "embeddingFile").action((x, conf) => conf.copy(embeddingFile = x)).text("embedding file, /path/to/vi/glove.6B.100d.txt")
      opt[Int]('w', "embeddingSize").action((x, conf) => conf.copy(embeddingSize = x)).text("embedding size, 100 or 200 or 300")
      opt[Int]('o', "encoderOutputSize").action((x, conf) => conf.copy(outputSize = x)).text("output size of the encoder")
      opt[Int]('n', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("maximum sequence length of a sentence")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Unit]('v', "verbose").action((x, conf) => conf.copy(verbose = true)).text("verbose mode")
    }
    parser.parse(args, ConfigNER()) match {
      case Some(config) =>
        val sparkConfig = Engine.createSparkConf()
          .setMaster(config.master)
          .set("spark.executor.memory", config.executorMemory)
          .setAppName("ner.NeuralTagger")
        val sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate()
        val sparkContext = sparkSession.sparkContext
        Engine.init

        println(Serialization.writePretty(config))

        val tagger = new NeuralTagger(sparkSession, config)
        val df = tagger.createDataFrame(config.dataPath)
        df.show()
        println(df.count())
        config.mode match {
          case "train" => tagger.train(df, df)
          case "eval" => 
          case "predict" => 
            val preprocessor = PipelineModel.load(Paths.get(config.modelPath, config.language, "gru").toString())
            val module = com.intel.analytics.bigdl.nn.Module.loadModule[Float](tagger.prefix + ".bigdl", tagger.prefix + ".bin")
            val model = new DLModel(module, featureSize = Array(config.maxSequenceLength))
            val prediction = tagger.predict(df, preprocessor, model)
            prediction.select("x", "z").show(false)
            prediction.printSchema()
            tagger.predict(config.dataPath, preprocessor, model)
        }
        sparkSession.stop()
      case None => 
    }
  }
}
