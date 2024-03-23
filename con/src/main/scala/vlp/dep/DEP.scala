package vlp.dep

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.models.{Model, Models}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.{ClassNLLCriterion, TimeDistributedMaskCriterion}
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.Trigger
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import scopt.OptionParser
import vlp.con.{ArgMaxLayer, Sequencer, TimeDistributedTop1Accuracy}

import java.nio.file.{Files, Paths, StandardOpenOption}

object DEP {

  /**
   * Linearize a graph into 4 sequences: Seq[word], Seq[PoS], Seq[labels], Seq[offsets].
   * @param graph a dependency graph
   * @param las labeled attachment score
   * @return a sequence of sequences.
   */
  private def linearize(graph: Graph, las: Boolean = false): Seq[Seq[String]] = {
    val tokens = graph.sentence.tokens.tail // remove the ROOT token at the beginning
    val words = tokens.map(_.word.toLowerCase()) // make all token lowercase
    val partsOfSpeech = tokens.map(_.partOfSpeech)
    val uPoS = tokens.map(_.universalPartOfSpeech)
    val labels = tokens.map(_.dependencyLabel)
    // compute the offset value to the head for each token
    val offsets = try {
      tokens.map { token =>
        val o = if (token.head.toInt == 0) 0 else token.head.toInt - token.id.toInt
        o.toString
      }
    } catch {
      case _: NumberFormatException =>
        print(graph)
        Seq.empty[String]
    } finally {
      Seq.empty[String]
    }
    val offsetLabels = if (las) {
      offsets.zip(labels).map(pair => pair._1 + ":" + pair._2)
    } else offsets
    Seq(words, partsOfSpeech, uPoS, labels, offsetLabels)
  }

  /**
   * Read graphs from a corpus, filter too long or too short graphs and convert them to a df.
   * @param spark Spark session
   * @param path path to a UD treebank
   * @param maxSeqLen maximum sequence length
   * @param las LAS or UAS
   * @return a data frame
   */
  private def readGraphs(spark: SparkSession, path: String, maxSeqLen: Int, las: Boolean = false): DataFrame = {
    // read graphs and remove too-long sentences
    val graphs = GraphReader.read(path).filter(_.sentence.tokens.size <= maxSeqLen).filter(_.sentence.tokens.size >= 5)
    // linearize the graph
    val xs = graphs.map { graph => Row(linearize(graph, las): _*) } // need to scroll out the parts with :_*
    val schema = StructType(Array(
      StructField("tokens", ArrayType(StringType, containsNull = true)),
      StructField("partsOfSpeech", ArrayType(StringType, containsNull = true)),
      StructField("uPoS", ArrayType(StringType, containsNull = true)),
      StructField("labels", ArrayType(StringType, containsNull = true)),
      StructField("offsets", ArrayType(StringType, containsNull = true))
    ))
    spark.createDataFrame(spark.sparkContext.parallelize(xs), schema)
  }

  /**
   * Create a preprocessing pipeline.
   *
   * @param df     a data frame
   * @param config configuration
   * @return a pipeline model
   */
  private def createPipeline(df: DataFrame, config: ConfigDEP): PipelineModel = {
    val vectorizerOffsets = new CountVectorizer().setInputCol("offsets").setOutputCol("off")
    val vectorizerTokens = new CountVectorizer().setInputCol("tokens").setOutputCol("tok").setVocabSize(config.maxVocabSize)
    val vectorizerPartsOfSpeech = new CountVectorizer().setInputCol("partsOfSpeech").setOutputCol("pos") // may use "uPoS" instead of "partsOfSpeech"
    val stages = Array(vectorizerOffsets, vectorizerTokens, vectorizerPartsOfSpeech)
    val pipeline = new Pipeline().setStages(stages)
    val model = pipeline.fit(df)
    model.write.overwrite().save(s"${config.modelPath}/${config.language}-pre")
    val ef = model.transform(df)
    ef.repartition(1).write.mode("overwrite").parquet(s"${config.modelPath}/${config.language}-dfs")
    model
  }

  /**
   * Evaluate a model on training data frame (uf) and validation data frame (vf) which have been preprocessed
   * by the training pipeline.
   * @param config config
   * @param uf training df
   * @param vf validation df
   * @param featureColName feature column name
   * @param offsetIndex map of index -> label
   * @param offsetMap map of label -> index (inverse of offsetIndex)
   */
  private def eval(config: ConfigDEP, uf: DataFrame, vf: DataFrame, featureColName: String, offsetIndex: Map[Int, String], offsetMap: Map[String, Int]): Seq[Double] = {
    val modelPath = s"${config.modelPath}/${config.language}-${config.modelType}"
    println(s"Loading model in the path: $modelPath...")
    val bigdl = Models.loadModel(modelPath)
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
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("y").setPredictionCol("z").setMetricName("accuracy")
    import spark.implicits._
    val scores = for (prediction <- predictions) yield {
      val zf = prediction.select("offsets", "z").map { row =>
        val os = row.getSeq[String](0).map(v => offsetMap.getOrElse(v, 1)) // gold offsets, indices of ["-4:nsubj",...]
        val p = row.getSeq[Float](1).take(os.size).map(v => offsetIndex(v.toInt)) // predicted offsets, indices of ["-5:nsubj", ...]
        // heuristic: make all out-of-bound indices to left most
        val p2 = p.map { v =>
          if (config.las) { // with dependency label, v is of value such as "-4:nsubj"
            val j = v.indexOf(":")
            val offset = v.substring(0, j)
            if (Math.abs(offset.toInt) >= os.length) {
              // wrong prediction, try to correct using a heuristic (assign to left most)
              val correct = "0:root"
              offsetMap.getOrElse(correct, 1)
            } else offsetMap.getOrElse(v, 1)
          } else { // without dependency label, v is of value such as "-4"
            if (Math.abs(v.toInt) >= os.length) 0 else v.toInt // previous experiments: use 0 instead of -1.
          }
        }
        (os, p2)
      }
      // show the prediction, each row is a graph
      zf.toDF("offsets", "prediction").show(5)
      // flatten the prediction, convert to double for evaluation using Spark lib
      val yz = zf.flatMap(p => p._2.map(_.toDouble).zip(p._1.map(_.toDouble))).toDF("z", "y")
      yz.show(15)
      evaluator.evaluate(yz)
    }
    scores
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[ConfigDEP](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 8g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/predict")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language (eng/ind/vie)")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('n', "maxSeqLength").action((x, conf) => conf.copy(maxSeqLen = x)).text("max sequence length")
      opt[Int]('j', "layers").action((x, conf) => conf.copy(layers = x)).text("number of RNN layers or Transformer blocks")
      opt[Int]('u', "heads").action((x, conf) => conf.copy(heads = x)).text("number of Transformer heads")
      opt[Int]('w', "tokenEmbeddingSize").action((x, conf) => conf.copy(tokenEmbeddingSize = x)).text("token embedding size")
      opt[Int]('h', "tokenHiddenSize").action((x, conf) => conf.copy(tokenHiddenSize = x)).text("encoder hidden size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 5E-3")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[String]('s', "scorePath").action((x, conf) => conf.copy(scorePath = x)).text("score path")
    }
    opts.parse(args, ConfigDEP()) match {
      case Some(config) =>
        val conf = new SparkConf().setAppName(getClass.getName).setMaster(config.master)
          .set("spark.driver.memory", config.driverMemory)
        // Creates or gets SparkContext with optimized configuration for BigDL performance.
        // The method will also initialize the BigDL engine.
        val sc = NNContext.initNNContext(conf)
        sc.setLogLevel("ERROR")
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        // determine the training and validation paths
        val (trainPath, validPath, testPath, gloveFile, numberbatchFile) = config.language match {
          case "eng" => ("dat/dep/eng/UD_English-EWT/en_ewt-ud-train.conllu",
            "dat/dep/eng/UD_English-EWT/en_ewt-ud-dev.conllu",
            "dat/dep/eng/UD_English-EWT/en_ewt-ud-test.conllu",
            "dat/emb/glove.6B.100d.vocab.txt",
            "dat/emb/numberbatch-en-19.08.vocab.txt"
        )
          case "ind" => ("dat/dep/ind/UD_Indonesian-GSD/id_gsd-ud-train.conllu",
            "dat/dep/ind/UD_Indonesian-GSD/id_gsd-ud-dev.conllu",
            "dat/dep/ind/UD_Indonesian-GSD/id_gsd-ud-test.conllu",
            "dat/emb/cc.id.300.vocab.vec",
            "dat/emb/numberbatch-id-19.08.vocab.txt"
        )
          case "vie" => ("dat/dep/vie/UD_Vietnamese-VTB/vi_vtb-ud-train.conllu",
            "dat/dep/vie/UD_Vietnamese-VTB/vi_vtb-ud-dev.conllu",
            "dat/dep/vie/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu",
            "dat/emb/cc.vi.300.vocab.vec",
            "dat/emb/numberbatch-vi-19.08.vocab.txt"
          )
          case _ =>
            println("Invalid language code!")
            ("", "", "", "", "")
        }
        // read the training the data set
        val df = readGraphs(spark, trainPath, config.maxSeqLen, config.las)
        // train a preprocessor and use it to transform training/dev data sets
        val preprocessor = createPipeline(df, config)
        val ef = preprocessor.transform(df)
        // read the validation data set
        val dfV = readGraphs(spark, validPath, config.maxSeqLen, config.las)
        val efV = preprocessor.transform(dfV)
        // read the test data set
        val dfW = readGraphs(spark, testPath, config.maxSeqLen, config.las)
        val efW = preprocessor.transform(dfW)
        efV.show(5)
        println("#(trainGraphs) = " + df.count())
        println("#(validGraphs) = " + dfV.count())
        println("#(testGraphs) = " + dfW.count())

        // array of offset labels, index -> label: offsets[i] is the label at index i:
        val offsets = preprocessor.stages(0).asInstanceOf[CountVectorizerModel].vocabulary
        println(offsets.mkString(", "))
        val offsetMap = offsets.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap // 1-based index for BigDL
        val offsetIndex = offsets.zipWithIndex.map(p => (p._2 + 1, p._1)).toMap
        val numOffsets = offsets.length
        val tokens = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
        val tokensMap = tokens.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap   // 1-based index for BigDL
        val numVocab = tokens.length
        val partsOfSpeech = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
        val partsOfSpeechMap = partsOfSpeech.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap // 1-based index for BigDL
        val numPartsOfSpeech = partsOfSpeech.length
        println("#(labels) = " + numOffsets)
        println(" #(vocab) = " + numVocab)
        println("   #(PoS) = " + numPartsOfSpeech)
        // extract token, pos and offset indices (start from 1 to use in BigDL)
        val tokenSequencer = new Sequencer(tokensMap, config.maxSeqLen, 0f).setInputCol("tokens").setOutputCol("t")
        val posSequencer = new Sequencer(partsOfSpeechMap, config.maxSeqLen, 0f).setInputCol("partsOfSpeech").setOutputCol("p")
        val offsetsSequencer = new Sequencer(offsetMap, config.maxSeqLen, -1f).setInputCol("offsets").setOutputCol("o")
        val gf = offsetsSequencer.transform(posSequencer.transform(tokenSequencer.transform(ef)))
        val gfV = offsetsSequencer.transform(posSequencer.transform(tokenSequencer.transform(efV)))
        val gfW = offsetsSequencer.transform(posSequencer.transform(tokenSequencer.transform(efW)))
        gfV.select("t", "o").show()

        // prepare train/valid/test data frame for each model type:
        val (uf, vf, wf) = config.modelType match {
          case "t" => (gf, gfV, gfW)
          case "tg" => (gf, gfV, gfW)
          case "tn" => (gf, gfV, gfW)
          case "t+p" | "tg+p" |  "tn+p" =>
            // assemble the two input vectors into one of double maxSeqLen (for use in a combined model)
            val assembler = new VectorAssembler().setInputCols(Array("t", "p")).setOutputCol("t+p")
            val hf = assembler.transform(gf)
            val hfV = assembler.transform(gfV)
            val hfW = assembler.transform(gfW)
            hfV.select("t+p", "o").show(5)
            (hf, hfV, hfW)
          case "b" =>
            // first, create a UDF g to make BERT input of size 4 x maxSeqLen
            import org.apache.spark.sql.functions._
            val g = udf((x: org.apache.spark.ml.linalg.Vector) => {
              val v = x.toArray // x is a dense vector (produced by a Sequencer)
              // token type, all are 0 (0 for sentence A, 1 for sentence B -- here we have only one sentence)
              val types: Array[Double] = Array.fill[Double](v.length)(0)
              // positions, start from 0
              val positions = Array.fill[Double](v.length)(0)
              for (j <- v.indices)
                positions(j) = j
              // attention mask with indices in [0, 1]
              // It's a mask to be used if the input sequence length is smaller than maxSeqLen
              // find the last non-zero element index
              var n = v.length - 1
              while (n >= 0 && v(n) == 0) n = n - 1
              val masks = Array.fill[Double](v.length)(1)
              // padded positions have a mask value of 0 (which are not attended to)
              for (j <- n + 1 until v.length) {
                masks(j) = 0
              }
              Vectors.dense(v ++ types ++ positions ++ masks)
            })
            // then transform the token index column
            val hf = gf.withColumn("tb", g(col("t")))
            val hfV = gfV.withColumn("tb", g(col("t")))
            val hfW = gfW.withColumn("tb", g(col("t")))
            (hf, hfV, hfW)
        }
        // create a BigDL model corresponding to a model type:
        val (bigdl, featureSize, labelSize, featureColName) = config.modelType  match {
          case "t" =>
            // 1. Sequential model with random token embeddings
            val bigdl = Sequential()
            bigdl.add(Embedding(numVocab + 1, config.tokenEmbeddingSize, inputLength = config.maxSeqLen))
            for (_ <- 1 to config.layers)
              bigdl.add(Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true)))
            bigdl.add(Dropout(0.5))
            bigdl.add(Dense(numOffsets, activation = "softmax"))
            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(config.maxSeqLen))
            bigdl.summary()
            (bigdl, featureSize, labelSize, "t")
          case "tg" =>
            // 1.1 Sequential model with pretrained token embeddings (GloVe)
            val bigdl = Sequential()
            bigdl.add(WordEmbeddingP(gloveFile, tokensMap, inputLength = config.maxSeqLen))
            for (_ <- 1 to config.layers)
              bigdl.add(Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true)))
            bigdl.add(Dropout(0.5))
            bigdl.add(Dense(numOffsets, activation = "softmax"))
            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(config.maxSeqLen))
            bigdl.summary()
            (bigdl, featureSize, labelSize, "t")
          case "tn" =>
            // 1.2 Sequential model with pretrained token embeddings (Numberbatch of ConceptNet)
            val bigdl = Sequential()
            bigdl.add(WordEmbeddingP(numberbatchFile, tokensMap, inputLength = config.maxSeqLen))
            for (_ <- 1 to config.layers)
              bigdl.add(Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true)))
            bigdl.add(Dropout(0.5))
            bigdl.add(Dense(numOffsets, activation = "softmax"))
            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen)), Array(config.maxSeqLen))
            bigdl.summary()
            (bigdl, featureSize, labelSize, "t")
          case "t+p"  | "tg+p" |  "tn+p" =>
            // A model for (token ++ partsOfSpeech) tensor
            val input = Input(inputShape = Shape(2*config.maxSeqLen), name = "input")
            val reshape = Reshape(targetShape = Array(2, config.maxSeqLen)).setName("reshape").inputs(input)
            val split = SplitTensor(1, 2).inputs(reshape)
            val token = SelectTable(0).setName("inputT").inputs(split)
            val inputT = Squeeze(1).inputs(token)
            val partsOfSpeech = SelectTable(1).setName("inputP").inputs(split)
            val inputP = Squeeze(1).inputs(partsOfSpeech)
            val embeddingT = config.modelType match {
              case "t+p" => Embedding(numVocab + 1, config.tokenEmbeddingSize).setName ("tokEmbedding").inputs(inputT)
              case "tg+p" => WordEmbeddingP(gloveFile, tokensMap, inputLength = config.maxSeqLen).setName("tokenEmbedding").inputs(inputT)
              case "tn+p" => WordEmbeddingP(numberbatchFile, tokensMap, inputLength = config.maxSeqLen).setName("tokenEmbedding").inputs(inputT)
            }
            val embeddingP = Embedding(numPartsOfSpeech + 1, config.partsOfSpeechEmbeddingSize).setName("posEmbedding").inputs(inputP)
            val merge = Merge.merge(inputs = List(embeddingT, embeddingP), mode = "concat")
            val rnn1 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-1")).inputs(merge)
            val rnn2 = Bidirectional(LSTM(outputDim = config.tokenHiddenSize, returnSequences = true).setName("LSTM-2")).inputs(rnn1)
            val dropout = Dropout(0.5).inputs(rnn2)
            val output = Dense(numOffsets, activation = "softmax").setName("dense").inputs(dropout)
            val bigdl = Model(input, output)
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
            val bert = BERT(vocab = numVocab + 1, hiddenSize = config.tokenEmbeddingSize, nBlock = config.layers, nHead = config.heads, maxPositionLen = config.maxSeqLen,
              intermediateSize = config.tokenHiddenSize, outputAllBlock = false).setName("bert")
            val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masksReshaped))
            val bertOutput = SelectTable(0).setName("firstBlock").inputs(bertNode)
            val dense = Dense(numOffsets).setName("dense").inputs(bertOutput)
            val output = SoftMax().setName("output").inputs(dense)
            val bigdl = Model(input, output)
//            val (featureSize, labelSize) = (Array(Array(config.maxSeqLen), Array(config.maxSeqLen), Array(config.maxSeqLen), Array(config.maxSeqLen)), Array(config.maxSeqLen))
            val (featureSize, labelSize) = (Array(Array(4*config.maxSeqLen)), Array(config.maxSeqLen))
            (bigdl, featureSize, labelSize, "tb")
        }

        def weights(): Tensor[Float] = {
          // compute label weights for the loss function
          import spark.implicits._
          val xf = df.select("offsets").flatMap(row => row.getSeq[String](0))
          val yf = xf.groupBy("value").count()
          val labelFreq = yf.select("value", "count").collect().map(row => (offsetMap(row.getString(0)), row.getLong(1)))
          val total = labelFreq.map(_._2).sum.toFloat
          val ws = labelFreq.map(p => (p._1, p._2 / total)).toMap
          val tensor = Tensor(ws.size)
          for (j <- ws.keys)
            tensor(j) = 1f/ws(j) // give higher weight to minority labels
          tensor
        }

        config.mode match {
          case "train" =>
            // create an estimator, it is necessary to set sizeAverage of ClassNLLCriterion to false in non-batch mode
            val estimator = if (config.weightedLoss)
              NNEstimator(bigdl, TimeDistributedMaskCriterion(ClassNLLCriterion(weights = weights(), sizeAverage = false, logProbAsInput = false, paddingValue = -1), paddingValue = -1), featureSize, labelSize)
            else NNEstimator(bigdl, TimeDistributedMaskCriterion(ClassNLLCriterion(sizeAverage = false, logProbAsInput = false, paddingValue = -1), paddingValue = -1), featureSize, labelSize)
            val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/dep/${config.language}")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/dep/${config.language}")
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
            bigdl.saveModel(s"${config.modelPath}/${config.language}-${config.modelType}", overWrite = true)
            // evaluate the model
            val scores = eval(config, uf, vf, featureColName, offsetIndex, offsetMap)
            val heads = if (config.modelType != "b") 0 else config.heads
            val result = f"\n${config.language};${config.modelType};${config.tokenEmbeddingSize};${config.tokenHiddenSize};${config.layers};$heads;${scores(0)}%.4g;${scores(1)}%.4g"
            println(result)
            Files.write(Paths.get(config.scorePath), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
          case "eval" =>
            // write score on training/test set
            val scores = eval(config, uf, wf, featureColName, offsetIndex, offsetMap)
            val heads = if (config.modelType != "b") 0 else config.heads
            val result = f"\n${config.language};${config.modelType};${config.tokenEmbeddingSize};${config.tokenHiddenSize};${config.layers};$heads;${scores(0)}%.4g;${scores(1)}%.4g"
            println(result)
            Files.write(Paths.get(config.scorePath), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
          case "predict" =>
            // train the model on the training set (uf) using the best hyper-parameters which was tuned on the validation set (vf)
            // and run prediction it on the test set (wf) to collect scores
            val estimator = if (config.weightedLoss)
              NNEstimator(bigdl, TimeDistributedMaskCriterion(ClassNLLCriterion(weights = weights(), sizeAverage = false, logProbAsInput = false, paddingValue = -1), paddingValue = -1), featureSize, labelSize)
            else
              NNEstimator(bigdl, TimeDistributedMaskCriterion(ClassNLLCriterion(sizeAverage = false, logProbAsInput = false, paddingValue = -1), paddingValue = -1), featureSize, labelSize)
            val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/dep/${config.language}")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/dep/${config.language}")
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
            val modelPath = s"${config.modelPath}/${config.language}-${config.modelType}"
            println(s"Save the model to $modelPath.")
            bigdl.saveModel(modelPath, overWrite = true)
            // evaluate the model on the training and test set
            val scores = eval(config, uf, wf, featureColName, offsetIndex, offsetMap)
            val heads = if (config.modelType != "b") 0 else config.heads
            val result = f"\n${config.language};${config.modelType};${config.tokenEmbeddingSize};${config.tokenHiddenSize};${config.layers};$heads;${scores(0)}%.4g;${scores(1)}%.4g"
            println(result)
            Files.write(Paths.get(config.scorePath), result.getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
          case "preprocess" =>
            val af = df.union(dfV).union(dfW)
            val preprocessor = createPipeline(af, config)
            val vocab = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.toSet
            println("#(vocab) = " + vocab.size)
            println(vocab)
        }
        spark.stop()
      case None => println("Invalid config!")
    }
  }
}
