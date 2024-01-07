package vlp.dep

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.keras.Sequential
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.models.{Model, Models}
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.{ClassNLLCriterion, TimeDistributedCriterion}
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.Trigger
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import vlp.con.TimeDistributedTop1Accuracy

case class ConfigDEP(
    master: String = "local[*]",
    totalCores: Int = 8,    // X
    executorCores: Int = 8, // Y ==> there are X/Y executors
    executorMemory: String = "8g", // Z
    driverMemory: String = "16g", // D
    mode: String = "eval",
    maxVocabSize: Int = 32768,
    tokenEmbeddingSize: Int = 32,
    partsOfSpeechEmbeddingSize: Int = 16,
    batchSize: Int = 128,
    maxSeqLen: Int = 30,
    hiddenSize: Int = 64,
    epochs: Int = 40,
    learningRate: Double = 5E-4,
    modelPath: String = "bin/dep/",
    trainPath: String = "dat/dep/eng/2.7/en_ewt-ud-train.conllu",
    validPath: String = "dat/dep/eng/2.7/en_ewt-ud-test.conllu",
    outputPath: String = "out/dep/",
    scorePath: String = "dat/dep/scores.json",
    modelType: String = "s",
)

object DEP {

  /**
   * Linearize a graph into 4 seqs: Seq[word], Seq[PoS], Seq[labels], Seq[offsets].
   * @param graph
   * @return a sequence of sequences.
   */
  def linearize(graph: Graph): Seq[Seq[String]] = {
    val tokens = graph.sentence.tokens.tail // remove the ROOT token at the beginning
    val words = tokens.map(_.word)
    val partsOfSpeech = tokens.map(_.partOfSpeech)
    val labels = tokens.map(_.dependencyLabel)
    val offsets = tokens.map(token => (token.head.toInt - token.id.toInt).toString) // offset from the head
    Seq(words, partsOfSpeech, labels, offsets)
  }

  def createPipeline(df: DataFrame, config: ConfigDEP) = {
    val vectorizerOffsets = new CountVectorizer().setInputCol("offsets").setOutputCol("off")
    val vectorizerToken = new CountVectorizer().setInputCol("tokens").setOutputCol("tok").setVocabSize(config.maxVocabSize)//.setMinDF(2)
    val vectorizerPoS = new CountVectorizer().setInputCol("partsOfSpeech").setOutputCol("pos")
    val pipeline = new Pipeline().setStages(Array(vectorizerOffsets, vectorizerToken, vectorizerPoS))
    val model = pipeline.fit(df)
    model.write.overwrite().save(config.modelPath)
    model
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[ConfigDEP](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 16g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('h', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("encoder hidden size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 5E-4")
      opt[String]('d', "trainPath").action((x, conf) => conf.copy(trainPath = x)).text("training data directory")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[String]('s', "scorePath").action((x, conf) => conf.copy(scorePath = x)).text("score path")
    }
    opts.parse(args, ConfigDEP()) match {
      case Some(config) =>
        val conf = new SparkConf().setAppName(getClass.getName).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        // Creates or gets SparkContext with optimized configuration for BigDL performance.
        // The method will also initialize the BigDL engine.
        val sc = NNContext.initNNContext(conf)
        sc.setLogLevel("ERROR")
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        // read graphs and remove too-long sentences
        val graphs = GraphReader.read(config.trainPath)
          .filter(_.sentence.tokens.size <= config.maxSeqLen)
          .filter(_.sentence.tokens.size >= 5)
        // linearize the graph
        val xs = graphs.map { graph => Row(linearize(graph):_*) } // need to scroll out the parts with :_*
        val schema = StructType(Array(
          StructField("tokens", ArrayType(StringType, true)),
          StructField("partsOfSpeech", ArrayType(StringType, true)),
          StructField("labels", ArrayType(StringType, true)),
          StructField("offsets", ArrayType(StringType, true))
        ))
        val df = spark.createDataFrame(sc.parallelize(xs), schema)
        df.show(5)
        println(df.count)
        val preprocessor = createPipeline(df, config)
        val ef = preprocessor.transform(df)
        val numOffsets = preprocessor.stages(0).asInstanceOf[CountVectorizerModel].vocabulary.size
        val numVocab = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.size
        val numPartsOfSpeech = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size

        import org.apache.spark.sql.functions._
        val f = udf((v: SparseVector) => v.indices.sorted.map(_ + 1f))
        val ff = ef.withColumn("tokIdx+1", f(col("tok")))
          .withColumn("posIdx+1", f(col("pos")))
          .withColumn("offIdx+1", f(col("off")))
        ff.show()
        ff.printSchema()
        // pad the input vectors to the same maximum length
        val padderT = new FeaturePadder(config.maxSeqLen, 0f).setInputCol("tokIdx+1").setOutputCol("t")
        val padderP = new FeaturePadder(config.maxSeqLen, 0f).setInputCol("posIdx+1").setOutputCol("p")
        // pad the output vector, use the paddingValue -1f
        val padderO = new FeaturePadder(config.maxSeqLen, -1f).setInputCol("offIdx+1").setOutputCol("o")
        val gf = padderO.transform(padderP.transform(padderT.transform(ff)))
        // assemble the two input vectors into one of double maxSeqLen (for use in a combined model)
        val hf = gf.withColumn("t+p", concat(col("t"), col("p")))
        hf.show()

        // create a BigDL model
        val (bigdl, featureSize, labelSize, featureColName) = if (config.modelType == "s") {
          // 1. Sequential
          val bigdl = Sequential()
          bigdl.add(InputLayer(inputShape = Shape(config.maxSeqLen)).setName("input"))
          bigdl.add(Embedding(numVocab + 1, config.tokenEmbeddingSize).setName("tokenEmbedding"))
          bigdl.add(Bidirectional(LSTM(outputDim = config.hiddenSize, returnSequences = true).setName("LSTM")))
          bigdl.add(Dense(numOffsets, activation = "softmax").setName("dense"))
          val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(config.maxSeqLen))
          val featureColName = "t"
          bigdl.summary()
          (bigdl, featureSize, labelSize, featureColName)
        } else {
          // 2. Model of multiple inputs
          val inputT = Input[Float](inputShape = Shape(config.maxSeqLen), "inputT")
          val inputP = Input[Float](inputShape = Shape(config.maxSeqLen), "inputP")
          val embeddingT = Embedding(numVocab + 1, config.tokenEmbeddingSize).setName("tokEmbedding").inputs(inputT)
          val embeddingP = Embedding(numPartsOfSpeech + 1, config.partsOfSpeechEmbeddingSize).setName("posEmbedding").inputs(inputP)
          val merge = Merge.merge(inputs = List(embeddingT, embeddingP), mode = "concat")
          val rnn = Bidirectional(LSTM(outputDim = config.hiddenSize, returnSequences = true).setName("LSTM")).inputs(merge)
          val output = Dense(numOffsets, activation = "softmax").setName("dense").inputs(rnn)
          val bigdl = Model[Float](Array(inputT, inputP), output)
          val (featureSize, labelSize) = (Array(Array(config.maxSeqLen), Array(config.maxSeqLen)), Array(config.maxSeqLen))
          val featureColName = "t+p"
          (bigdl, featureSize, labelSize, featureColName)
        }
        // create an estimator: use either gf or hf
        val estimator = NNEstimator(bigdl, TimeDistributedCriterion(ClassNLLCriterion(logProbAsInput = false), sizeAverage = true), featureSize, labelSize)
        val trainingSummary = TrainSummary(appName = config.modelType, logDir = "sum/dep/")
        val validationSummary = ValidationSummary(appName = config.modelType, logDir = "sum/dep/")
        estimator.setLabelCol("o").setFeaturesCol(featureColName)
          .setBatchSize(config.batchSize)
          .setOptimMethod(new Adam(config.learningRate))
          .setMaxEpoch(config.epochs)
          .setTrainSummary(trainingSummary)
          .setValidationSummary(validationSummary)
          .setValidation(Trigger.everyEpoch, hf, Array(new TimeDistributedTop1Accuracy(paddingValue = -1)), config.batchSize)
        estimator.fit(hf)

        spark.stop()
      case None =>

    }
  }
}
