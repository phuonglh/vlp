package vlp.dep

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.models.{Model, Models}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.{ClassNLLCriterion, TimeDistributedCriterion, TimeDistributedMaskCriterion}
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.Trigger
import com.intel.analytics.bigdl.dllib.tensor.{DenseTensorMath, Tensor}
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import scopt.OptionParser
import vlp.con.{ArgMaxLayer, Sequencer, TimeDistributedTop1Accuracy}

case class ConfigDEP(
    master: String = "local[*]",
    totalCores: Int = 8,    // X
    executorCores: Int = 4, // Y ==> there are X/Y executors
    executorMemory: String = "4g", // Z
    driverMemory: String = "16g", // D
    mode: String = "eval",
    maxVocabSize: Int = 32768,
    tokenEmbeddingSize: Int = 32, // 100
    partsOfSpeechEmbeddingSize: Int = 8, // 25
    layers: Int = 2, // number of LSTM layers or Transformer blocks
    batchSize: Int = 128,
    maxSeqLen: Int = 10,
    hiddenSize: Int = 32,
    epochs: Int = 100,
    learningRate: Double = 5E-3,
    modelPath: String = "bin/dep/eng",
    trainPath: String = "dat/dep/UD_English-EWT/en_ewt-ud-dev.conllu",
    validPath: String = "dat/dep/UD_English-EWT/en_ewt-ud-dev.conllu",
    testPath: String = "dat/dep/UD_English-EWT/en_ewt-ud-test.conllu",
    outputPath: String = "out/dep/",
    scorePath: String = "dat/dep/scores.json",
    modelType: String = "t+c", // [t, t+c, t+p, b]
    useCharacter: Boolean = true,
    maxCharLen: Int = 13,
    charEmbeddingSize: Int = 16,
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
    val uPoS = tokens.map(_.universalPartOfSpeech)
    val labels = tokens.map(_.dependencyLabel)
    // compute the offset value to the head for each token
    val offsets = tokens.map { token =>
      val o = if (token.head.toInt == 0) 0 else (token.head.toInt - token.id.toInt)
      o.toString
    }
    Seq(words, partsOfSpeech, uPoS, labels, offsets)
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
      StructField("uPoS", ArrayType(StringType, true)),
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
    val vectorizerTokens = new CountVectorizer().setInputCol("tokens").setOutputCol("tok").setVocabSize(config.maxVocabSize)
    val vectorizerPartsOfSpeech = new CountVectorizer().setInputCol("partsOfSpeech").setOutputCol("pos") // may use uPoS instead of partsOfSpeech
    val stages = if (!config.useCharacter) {
      Array(vectorizerOffsets, vectorizerTokens, vectorizerPartsOfSpeech)
    } else {
      val sequencerChars = new CharacterSequencer(config.maxCharLen).setInputCol("tokens").setOutputCol("chars")
      val vectorizerChars = new CountVectorizer().setInputCol("chars").setOutputCol("char")
      Array(vectorizerOffsets, vectorizerTokens, vectorizerPartsOfSpeech, sequencerChars, vectorizerChars)
    }
    val pipeline = new Pipeline().setStages(stages)
    val model = pipeline.fit(df)
    model.write.overwrite().save(config.modelPath + "-pre")
//    val ef = model.transform(df)
//    ef.repartition(1).write.mode("overwrite").parquet(config.modelPath + "-dfs")
    model
  }

  /**
   * Evaluate a model on training data frame (uf) and validation data frame (vf) which have been preprocessed
   * by the training pipeline.
   * @param config
   * @param uf
   * @param vf
   * @param featureColName
   * @param labels
   */
  private def eval(config: ConfigDEP, uf: DataFrame, vf: DataFrame, featureColName: String, labels: Array[String]) = {
    val bigdl = Models.loadModel(config.modelPath + "-" + config.modelType)
    // create a sequential model and add a custom ArgMax layer at the end of the model
    val sequential = Sequential()
    sequential.add(bigdl)
    // bigdl produces 3-d output results (including batch dimension), we need to convert it to 2-d results.
    sequential.add(ArgMaxLayer())
    // run prediction on the training set and validation set
    val predictions = Array(
      sequential.predict(uf, featureCols = Array(featureColName), predictionCol = "z"),
      sequential.predict(vf, featureCols = Array(featureColName), predictionCol = "z")
    )
    sequential.summary()
    val spark = SparkSession.getActiveSession.get
    import spark.implicits._
    for (prediction <- predictions) {
      val zf = prediction.select("offsets", "z").map { row =>
        val o = row.getSeq[String](0)
        val p = row.getSeq[Float](1).take(o.size)
        (o, p.map(v => labels(v.toInt - 1))) // convert 1-based index to offset label
      }.toDF("offsets", "prediction")
      zf.show(15)
    }
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
      opt[Int]('n', "maxSeqLength").action((x, conf) => conf.copy(maxSeqLen = x)).text("max sequence length")
      opt[Int]('j', "layers").action((x, conf) => conf.copy(layers = x)).text("number of RNN layers or Transformer blocks")
      opt[Int]('w', "tokenEmbeddingSize").action((x, conf) => conf.copy(tokenEmbeddingSize = x)).text("token embedding size")
      opt[Int]('h', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("encoder hidden size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 5E-3")
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

        // array of offset labels, index -> label: offsets[i] is the label at index i:
        val offsets = preprocessor.stages(0).asInstanceOf[CountVectorizerModel].vocabulary
        val offsetsMap = offsets.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap // 1-based index for BigDL
        val numOffsets = offsets.length
        val tokens = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
        val tokensMap = tokens.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
        val numVocab = tokens.length
        val partsOfSpeech = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
        val partsOfSpeechMap = partsOfSpeech.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
        val numPartsOfSpeech = partsOfSpeech.length
        println("#(labels) = " + numOffsets)
        println(" #(vocab) = " + numVocab)
        println("   #(PoS) = " + numPartsOfSpeech)
        // extract token, pos and offset indices (start from 1 to use in BigDL)
        val tokenSequencer = new Sequencer(tokensMap, config.maxSeqLen, 0f).setInputCol("tokens").setOutputCol("t")
        val posSequencer = new Sequencer(partsOfSpeechMap, config.maxSeqLen, 0f).setInputCol("partsOfSpeech").setOutputCol("p")
        val offsetsSequencer = new Sequencer(offsetsMap, config.maxSeqLen, -1f).setInputCol("offsets").setOutputCol("o")
        val gf = offsetsSequencer.transform(posSequencer.transform(tokenSequencer.transform(ef)))
        val gfV = offsetsSequencer.transform(posSequencer.transform(tokenSequencer.transform(efV)))
        gfV.select("t", "o").show()

        var numChars = 0
        val (pf, pfV) = if (config.useCharacter) {
          val characters = preprocessor.stages(4).asInstanceOf[CountVectorizerModel].vocabulary
          val characterMap = characters.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
          numChars = characters.length
          println(" #(chars) = " + numChars)
          println(characters.mkString(", "))
          val charSequencer = new Sequencer(characterMap, config.maxSeqLen * config.maxCharLen, 1f).setInputCol("chars").setOutputCol("c")
          (charSequencer.transform(gf), charSequencer.transform(gfV))
        } else (gf, gfV)

        // prepare train/valid data frame for each model type:
        val (uf, vf) = config.modelType match {
          case "t" => (gf, gfV)
          case "t+c" =>
            // assemble the two input vectors into one
            val assembler = new VectorAssembler().setInputCols(Array("t", "c")).setOutputCol("t+c")
            val hf = assembler.transform(pf)
            val hfV = assembler.transform(pfV)
            hf.select("c").show(3, false)
            (hf, hfV)
          case "t+p" =>
            // assemble the two input vectors into one of double maxSeqLen (for use in a combined model)
            val assembler = new VectorAssembler().setInputCols(Array("t", "p")).setOutputCol("t+p")
            val hf = assembler.transform(gf)
            val hfV = assembler.transform(gfV)
            hfV.select("t+p", "o").show(5)
            (hf, hfV)
          case "b" =>
            // first, create a UDF g to make BERT input of size 4 x maxSeqLen
            import org.apache.spark.sql.functions._
            val g = udf((x: org.apache.spark.ml.linalg.Vector) => {
              val v = x.toArray
              // token type, all are 0 (0 for sentence A, 1 for sentence B -- here we have only one sentence)
              val types: Array[Double] = Array.fill[Double](v.length)(0)
              // positions, start from 0
              val positions = Array.fill[Double](v.length)(0)
              for (j <- 0 until v.length)
                positions(j) = j
              // attention mask with indices in [0, 1]
              // It's a mask to be used if the input sequence length is smaller than maxSeqLen
              val i = v.indexOf(0) // 0 is the padded value (of tokens and partsOfSpeech)
              val n = if (i >= 0) i else v.length // the actual length of the un-padded sequence
              val masks = Array.fill[Double](v.length)(1)
              // padded positions have a mask value of 0
              for (j <- n until v.length) {
                masks(j) = 0
              }
              Vectors.dense(v ++ types ++ positions ++ masks)
            })
            // then transform the token index column
            val hf = gf.withColumn("tb", g(col("t")))
            val hfV = gfV.withColumn("tb", g(col("t")))
            (hf, hfV)
        }
        // create a BigDL model corresponding to a model type:
        val (bigdl, featureSize, labelSize, featureColName) = config.modelType  match {
          case "t" =>
            // 1. Sequential model with tokens only
            val bigdl = Sequential()
            bigdl.add(Embedding(numVocab + 1, config.tokenEmbeddingSize, inputLength = config.maxSeqLen))
            for (_ <- 1 to config.layers)
              bigdl.add(Bidirectional(LSTM(outputDim = config.hiddenSize, returnSequences = true)))
            bigdl.add(Dropout(0.5))
            bigdl.add(Dense(numOffsets, activation = "softmax"))
            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(config.maxSeqLen))
            bigdl.summary()
            (bigdl, featureSize, labelSize, "t")
          case "t+c" =>
            // NN-style layers (not Keras-style layer)
            val sequential = com.intel.analytics.bigdl.dllib.nn.Sequential()
            val lookup = com.intel.analytics.bigdl.dllib.nn.LookupTable(numChars + 1, config.charEmbeddingSize, paddingValue = 1f) // Note: paddingValue > 0
            sequential.add(lookup)
            val reshape = com.intel.analytics.bigdl.dllib.nn.Reshape(Array(config.maxSeqLen, config.maxCharLen, config.charEmbeddingSize))
            sequential.add(reshape)
            val splitTensor = com.intel.analytics.bigdl.dllib.nn.SplitTable(1, -1)
            sequential.add(splitTensor)
            val lstm = com.intel.analytics.bigdl.dllib.nn.LSTM(config.charEmbeddingSize, config.charEmbeddingSize)
            val mapTable = com.intel.analytics.bigdl.dllib.nn.MapTable(lstm)
            sequential.add(mapTable)
            val joinTable = com.intel.analytics.bigdl.dllib.nn.JoinTable(2, 2)
            sequential.add(joinTable)
//            val dense = com.intel.analytics.bigdl.dllib.nn.Linear(config.tokenEmbeddingSize + config.charEmbeddingSize, numOffsets)
            val dense = com.intel.analytics.bigdl.dllib.nn.Linear(config.charEmbeddingSize, numOffsets)
            sequential.add(dense)
            sequential.add(com.intel.analytics.bigdl.dllib.nn.SoftMax())
            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen * config.maxCharLen)), Array(config.maxSeqLen))
            println(sequential)
            (sequential, featureSize, labelSize, "c")
//          case "t+c" =>
//            // 1c. Sequential model with tokens and characters
//            val inputT = Input[Float](inputShape = Shape(config.maxSeqLen), name = "inputT")
//            val inputC = Input[Float](inputShape = Shape(config.maxSeqLen * config.maxCharLen), name = "inputC")
//            // token LSTM
//            val embeddingT = Embedding(numVocab + 1, config.tokenEmbeddingSize).setName("embeddingT").inputs(inputT)
//            val lstmT = LSTM(outputDim = config.hiddenSize, returnSequences = true).inputs(embeddingT)
//            // character LSTM
//            val embeddingC = Embedding(numChars + 1, config.charEmbeddingSize).setName("embeddingC").inputs(inputC)
//            val reshapeC1 = Reshape(targetShape = Array(config.maxSeqLen, config.maxCharLen, -1)).setName("reshapeC1").inputs(embeddingC)
//
//            val seq = Sequential()
//            val reshapeC2 = Reshape(targetShape = Array(config.maxCharLen, -1))
//            seq.add(reshapeC2)
//            val lstmC = LSTM(outputDim = config.charEmbeddingSize).setName("lstmC")
//            seq.add(lstmC)
//
//            // merge two LSTMs (concat mode)
//            val merge = Merge.merge(List(lstmT, seq.inputs(reshapeC1)), mode = "concat")
//            val output = Dense(numOffsets, activation = "softmax").inputs(merge)
//            val bigdl = Model[Float](Array(inputT, inputC), output)
//            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen), Array(config.maxSeqLen * config.maxCharLen)), Array(config.maxSeqLen))
//            println(bigdl.summary())
//            (bigdl, featureSize, labelSize, "t+c")
          case "t+p" =>
            // 2. A model for (token ++ partsOfSpeech) tensor
            val input = Input[Float](inputShape = Shape(2*config.maxSeqLen), name = "input")
            val reshape = Reshape(targetShape = Array(2, config.maxSeqLen)).setName("reshape").inputs(input)
            val split = SplitTensor(1, 2).inputs(reshape)
            val token = SelectTable(0).setName("inputT").inputs(split)
            val inputT = Squeeze(1).inputs(token)
            val partsOfSpeech = SelectTable(1).setName("inputP").inputs(split)
            val inputP = Squeeze(1).inputs(partsOfSpeech)
            // the above layers can be replaced by two following inputs, but then the input is supposed to be a table
//            val inputT = Input[Float](inputShape = Shape(config.maxSeqLen), "inputT")
//            val inputP = Input[Float](inputShape = Shape(config.maxSeqLen), "inputP")
            val embeddingT = Embedding(numVocab + 1, config.tokenEmbeddingSize).setName("tokEmbedding").inputs(inputT)
            val embeddingP = Embedding(numPartsOfSpeech + 1, config.partsOfSpeechEmbeddingSize).setName("posEmbedding").inputs(inputP)
            val merge = Merge.merge(inputs = List(embeddingT, embeddingP), mode = "concat")
            val rnn1 = Bidirectional(LSTM(outputDim = config.hiddenSize, returnSequences = true).setName("LSTM-1")).inputs(merge)
            val rnn2 = Bidirectional(LSTM(outputDim = config.hiddenSize, returnSequences = true).setName("LSTM-2")).inputs(rnn1)
            val dropout = Dropout(0.5).inputs(rnn2)
            val output = Dense(numOffsets, activation = "softmax").setName("dense").inputs(dropout)
            val bigdl = Model[Float](input, output)
            val (featureSize, labelSize) = (Array(Array(2*config.maxSeqLen)), Array(config.maxSeqLen))
            (bigdl, featureSize, labelSize, "t+p")
          case "b" =>
            // 4. BERT model using one input of 4*maxSeqLen elements
            val input = Input(inputShape = Shape(4*config.maxSeqLen), name = "input")
            val reshape = Reshape(targetShape = Array(4, config.maxSeqLen)).inputs(input)
            val split = SplitTensor(1, 4).inputs(reshape)
            val selectIds = SelectTable(0).setName("inputId").inputs(split)
            val inputIds = Squeeze(1).inputs(selectIds)
            val selectSegments = SelectTable(1).setName("segmentId").inputs(split)
            val segmentIds = Squeeze(1).inputs(selectSegments)
            val selectPositions = SelectTable(2).setName("positionId").inputs(split)
            val positionIds = Squeeze(1).inputs(selectPositions)
            val selectMasks = SelectTable(3).setName("masks").inputs(split)
            val masksReshaped = Reshape(targetShape = Array(1, 1, config.maxSeqLen)).setName("mask").inputs(selectMasks)
//            val inputIds = Input(inputShape = Shape(config.maxSeqLen), "inputIds")
//            val segmentIds = Input(inputShape = Shape(config.maxSeqLen), "segmentIds")
//            val positionIds = Input(inputShape = Shape(config.maxSeqLen), "positionIds")
//            val masks = Input(inputShape = Shape(config.maxSeqLen), "masks")
//            val masksReshaped = Reshape(targetShape = Array(1, 1, config.maxSeqLen)).setName("reshape").inputs(masks)
            val bert = BERT(vocab = numVocab + 1, hiddenSize = config.tokenEmbeddingSize, nBlock = config.layers, nHead = 2, maxPositionLen = config.maxSeqLen,
              intermediateSize = config.hiddenSize, outputAllBlock = false).setName("bert")
            val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masksReshaped))
            val bertOutput = SelectTable(0).setName("firstBlock").inputs(bertNode)
            val dense = Dense(numOffsets).setName("dense").inputs(bertOutput)
            val output = SoftMax().setName("output").inputs(dense)
            val bigdl = Model(input, output)
//            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen), Array(config.maxSeqLen), Array(config.maxSeqLen), Array(config.maxSeqLen)), Array(config.maxSeqLen))
            val (featureSize, labelSize) = (Array(Array(4*config.maxSeqLen)), Array(config.maxSeqLen))
            (bigdl, featureSize, labelSize, "tb")
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
            val tensor = Tensor(ws.length)
            for (j <- ws.indices)
              tensor(j+1) = ws(j)
            // create an estimator
            TimeDistributedMaskCriterion
            // it is necessary to set sizeAverage of ClassNLLCriterion to false in non-batch mode
            val estimator = NNEstimator(bigdl, TimeDistributedMaskCriterion(ClassNLLCriterion(weights = tensor, sizeAverage = false, logProbAsInput = false, paddingValue = -1), paddingValue = -1), featureSize, labelSize)
            val trainingSummary = TrainSummary(appName = config.modelType, logDir = "sum/dep/")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = "sum/dep/")
            estimator.setLabelCol("o").setFeaturesCol(featureColName)
              .setBatchSize(config.batchSize)
              .setOptimMethod(new Adam(config.learningRate))
              .setMaxEpoch(config.epochs)
              .setTrainSummary(trainingSummary)
              .setValidationSummary(validationSummary)
              .setValidation(Trigger.everyEpoch, vf, Array(new TimeDistributedTop1Accuracy(-1)), config.batchSize)
            // train
            estimator.fit(uf)
            // save the model
            bigdl.saveModel(config.modelPath + s"-${config.modelType}", overWrite = true)
            // evaluate the model
            eval(config, uf, vf, featureColName, offsets)
          case "eval" =>
            eval(config, uf, vf, featureColName, offsets)
          case "pre" =>
            ef.select("chars").show(5,false)
        }
        spark.stop()
      case None =>

    }
  }
}
