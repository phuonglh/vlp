package vlp.dep

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.models.{Model, Models}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.{ClassNLLCriterion, TimeDistributedCriterion}
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.Trigger
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.linalg.{SparseVector, Vectors}
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import scopt.OptionParser
import vlp.con.{ArgMaxLayer, TimeDistributedTop1Accuracy}

case class ConfigDEP(
    master: String = "local[*]",
    totalCores: Int = 8,    // X
    executorCores: Int = 4, // Y ==> there are X/Y executors
    executorMemory: String = "4g", // Z
    driverMemory: String = "16g", // D
    mode: String = "eval",
    maxVocabSize: Int = 32768,
    tokenEmbeddingSize: Int = 32,
    partsOfSpeechEmbeddingSize: Int = 16,
    batchSize: Int = 128,
    maxSeqLen: Int = 30,
    hiddenSize: Int = 64,
    epochs: Int = 50,
    learningRate: Double = 5E-4,
    modelPath: String = "bin/dep/eng",
    trainPath: String = "dat/dep/UD_English-EWT/en_ewt-ud-train.conllu",
    validPath: String = "dat/dep/UD_English-EWT/en_ewt-ud-dev.conllu",
    testPath: String = "dat/dep/UD_English-EWT/en_ewt-ud-test.conllu",
    outputPath: String = "out/dep/",
    scorePath: String = "dat/dep/scores.json",
    modelType: String = "s", // [s, g]
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
    // compute the offset value to the head for each token
    val offsets = tokens.map { token =>
      val o = if (token.head.toInt == 0) 0 else (token.head.toInt - token.id.toInt)
      o.toString
    }
    Seq(words, partsOfSpeech, labels, offsets)
  }

  /**
   * Read graphs from a corpus, filter too long or too short graphs and convert them to a df.
   * @param spark
   * @param path
   * @param maxSeqLen
   * @return
   */
  def readGraphs(spark: SparkSession, path: String, maxSeqLen: Int) = {
    // read graphs and remove too-long sentences
    val graphs = GraphReader.read(path).filter(_.sentence.tokens.size <= maxSeqLen).filter(_.sentence.tokens.size >= 5)
    // linearize the graph
    val xs = graphs.map { graph => Row(linearize(graph): _*) } // need to scroll out the parts with :_*
    val schema = StructType(Array(
      StructField("tokens", ArrayType(StringType, true)),
      StructField("partsOfSpeech", ArrayType(StringType, true)),
      StructField("labels", ArrayType(StringType, true)),
      StructField("offsets", ArrayType(StringType, true))
    ))
    spark.createDataFrame(spark.sparkContext.parallelize(xs), schema)
  }

  /**
   * Create a preprocessing pipeline
   * @param df
   * @param config
   * @return a pipeline model
   */
  private def createPipeline(df: DataFrame, config: ConfigDEP) = {
    val vectorizerOffsets = new CountVectorizer().setInputCol("offsets").setOutputCol("off")
    val vectorizerToken = new CountVectorizer().setInputCol("tokens").setOutputCol("tok").setVocabSize(config.maxVocabSize)
    val vectorizerPoS = new CountVectorizer().setInputCol("partsOfSpeech").setOutputCol("pos")
    val pipeline = new Pipeline().setStages(Array(vectorizerOffsets, vectorizerToken, vectorizerPoS))
    val model = pipeline.fit(df)
    model.write.overwrite().save(config.modelPath + "-pre")
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

        // read training data, train a preprocessor and use it to transform training/dev data sets
        val df = readGraphs(spark, config.trainPath, config.maxSeqLen)
        val preprocessor = createPipeline(df, config)
        val ef = preprocessor.transform(df)
        // read validation data set
        val dfV = readGraphs(spark, config.validPath, config.maxSeqLen)
        val efV = preprocessor.transform(dfV)
        efV.show(5)
        println("#(trainGraphs) = " + df.count())
        println("#(validGraphs) = " + dfV.count())

        val offsets = preprocessor.stages(0).asInstanceOf[CountVectorizerModel].vocabulary
        val offsetsMap = offsets.zipWithIndex.toMap
        val numOffsets = offsetsMap.size
        val numVocab = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.size
        val numPartsOfSpeech = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size
        // extract token, pos and offset indices (plus 1 to use in BigDL)
        import org.apache.spark.sql.functions._
        val f = udf((v: SparseVector) => v.indices.sorted.map(_ + 1f))
        val ff = ef.withColumn("tokIdx+1", f(col("tok")))
          .withColumn("posIdx+1", f(col("pos")))
          .withColumn("offIdx+1", f(col("off")))
        val ffV = efV.withColumn("tokIdx+1", f(col("tok")))
          .withColumn("posIdx+1", f(col("pos")))
          .withColumn("offIdx+1", f(col("off")))

        // pad the input vectors to the same maximum length
        val padderT = new FeaturePadder(config.maxSeqLen, 0f).setInputCol("tokIdx+1").setOutputCol("t")
        val padderP = new FeaturePadder(config.maxSeqLen, 0f).setInputCol("posIdx+1").setOutputCol("p")
        // pad the output vector, use the paddingValue -1f
        val padderO = new FeaturePadder(config.maxSeqLen, -1f).setInputCol("offIdx+1").setOutputCol("o")
        val gf = padderO.transform(padderP.transform(padderT.transform(ff)))
        val gfV = padderO.transform(padderP.transform(padderT.transform(ffV)))

        val (uf, vf) = config.modelType match {
          case "s" => (gf, gfV)
          case "t" => (gf, gfV)
          case "g" =>
            // assemble the two input vectors into one of double maxSeqLen (for use in a combined model)
            val hf = gf.withColumn("t+p", concat(col("t"), col("p")))
            val hfV = gfV.withColumn("t+p", concat(col("t"), col("p")))
            hfV.select("t").show(5, false)
            hfV.select("t+p", "o").show(5)
            (hf, hfV)
          case "b" =>
            // first, create a UDF g to make BERT input of size 4 x maxSeqLen
            val g = udf((v: Seq[Float]) => {
              // token type, all are 0 (0 for sentence A, 1 for sentence B -- here we have only one sentence)
              val types = Array.fill[Double](v.size)(0)
              // positions, start from 0
              val positions = Array.fill[Double](v.size)(0)
              for (j <- 0 until v.size)
                positions(j) = j
              // attention mask with indices in [0, 1]
              // It's a mask to be used if the input sequence length is smaller than maxSeqLen
              val i = v.indexOf(0f) // 0f is the padded value (of tokens and partsOfSpeech)
              val n = if (i >= 0) i else v.size // the actual length of the un-padded sequence
              val masks = Array.fill[Double](v.size)(1)
              // padded positions have a mask value of 0
              for (j <- n until v.size) {
                masks(j) = 0
              }
              Vectors.dense(v.toArray.map(_.toDouble) ++ types ++ positions ++ masks)
            })
            // then transform the token indext column
            val hf = gf.withColumn("tb", g(col("t")))
            val hfV = gfV.withColumn("tb", g(col("t")))
            (hf, hfV)
        }
        // create a BigDL model
        val (bigdl, featureSize, labelSize, featureColName) = config.modelType  match {
          case "s" =>
            // 1. Sequential model for a single input
            val bigdl = Sequential()
            bigdl.add(InputLayer(inputShape = Shape(config.maxSeqLen)).setName("input"))
            bigdl.add(Embedding(numVocab + 1, config.tokenEmbeddingSize).setName("tokenEmbedding"))
            bigdl.add(Bidirectional(LSTM(outputDim = config.hiddenSize, returnSequences = true).setName("LSTM")))
            bigdl.add(Dense(numOffsets, activation = "softmax").setName("dense"))
            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(config.maxSeqLen))
            val featureColName = "t"
            bigdl.summary()
            (bigdl, featureSize, labelSize, featureColName)
          case "g" =>
            // 2. Graph model for multiple inputs (token + partsOfSpeech)
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
          case "b" =>
            // 4. BERT model for a single input
            val inputIds = Input(inputShape = Shape(config.maxSeqLen), "inputIds")
            val segmentIds = Input(inputShape = Shape(config.maxSeqLen), "segmentIds")
            val positionIds = Input(inputShape = Shape(config.maxSeqLen), "positionIds")
            val masks = Input(inputShape = Shape(config.maxSeqLen), "masks")
            val masksReshaped = Reshape(targetShape = Array(1, 1, config.maxSeqLen)).setName("reshape").inputs(masks)
            val bert = BERT(vocab = numVocab + 1, hiddenSize = config.tokenEmbeddingSize, nBlock = 1, nHead = 4, maxPositionLen = config.maxSeqLen,
              intermediateSize = config.hiddenSize, outputAllBlock = false).setName("bert")
            val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masksReshaped))
            val bertOutput = SelectTable(0).setName("firstBlock").inputs(bertNode)
            val dense = Dense(numOffsets).setName("dense").inputs(bertOutput)
            val output = SoftMax().setName("output").inputs(dense)
            val bigdl = Model(Array(inputIds, segmentIds, positionIds, masks), output)
            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen), Array(config.maxSeqLen), Array(config.maxSeqLen), Array(config.maxSeqLen)), Array(config.maxSeqLen))
            val featureColName = "tb"
            (bigdl, featureSize, labelSize, featureColName)
        }
        config.mode match {
          case "train" =>
            // compute label weights for the loss function
            import spark.implicits._
            val xf = df.select("offsets").flatMap(row => row.getSeq[String](0))
            val yf = xf.groupBy("value").count()
            val labelFreq = yf.select("value", "count").collect().map(row => (offsetsMap(row.getString(0)), row.getLong(1)))
            val total = labelFreq.map(_._2).sum.toFloat
            val ws = labelFreq.map(_._2 / total)
            val tensor = Tensor[Float](ws.size)
            for (j <- 0 until ws.size)
              tensor(j+1) = ws(j)
            // create an estimator
            val estimator = NNEstimator(bigdl, TimeDistributedCriterion(ClassNLLCriterion(weights = tensor, logProbAsInput = false), sizeAverage = true), featureSize, labelSize)
            val trainingSummary = TrainSummary(appName = config.modelType, logDir = "sum/dep/")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = "sum/dep/")
            estimator.setLabelCol("o").setFeaturesCol(featureColName)
              .setBatchSize(config.batchSize)
              .setOptimMethod(new Adam(config.learningRate))
              .setMaxEpoch(config.epochs)
              .setTrainSummary(trainingSummary)
              .setValidationSummary(validationSummary)
              .setValidation(Trigger.everyEpoch, vf, Array(new TimeDistributedTop1Accuracy(paddingValue = -1)), config.batchSize)
            // train
            estimator.fit(uf)
            // save the model
          bigdl.saveModel(config.modelPath, overWrite = true)
          case "eval" =>
            val bigdl = Models.loadModel(config.modelPath)
            val sequential = bigdl.asInstanceOf[Sequential[Float]]
            // bigdl produces 3-d output results (including batch dimension), we need to convert it to 2-d results.
            sequential.add(ArgMaxLayer())
            sequential.summary()
            // run prediction
            val prediction = bigdl.predict(vf, featureCols = Array(featureColName), predictionCol = "z")
            import spark.implicits._
            val zf = prediction.select("offsets", "z").map { row =>
              val o = row.getSeq[String](0)
              val p = row.getSeq[Float](1).take(o.size)
              (o, p.map(v => offsets(v.toInt-1)))
            }.toDF("offsets", "prediction")
            zf.show(10, false)
        }
        spark.stop()
      case None =>

    }
  }
}
