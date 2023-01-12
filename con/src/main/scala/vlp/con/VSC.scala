package vlp.con

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.keras.models.Models
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.TimeDistributedCriterion
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.models._
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.optim.Loss
import com.intel.analytics.bigdl.dllib.utils.Engine

import org.apache.spark.SparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature._
import org.apache.spark.ml._
import org.apache.spark.ml.linalg.Vector

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

import org.apache.log4j.{Logger, Level}
import org.slf4j.LoggerFactory
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.nnframes.{NNModel, NNEstimator}
import com.intel.analytics.bigdl.dllib.optim.Trigger
import com.intel.analytics.bigdl.dllib.nn.{TimeDistributedCriterion, ClassNLLCriterion}


object VSC {

  /**
    * Reads an input text file and creates a data frame of two columns "x, y", where 
    * "x" are input token sequences and "y" are corresponding label sequences. The text file 
    * has a format of line-pair oriented: (y_i, x_i).
    *
    * @param sc a Spark context
    * @param config
    */
  def readData(sc: SparkContext, config: Config): DataFrame = {
    val spark = SparkSession.builder.getOrCreate()
    import spark.implicits._
    val df = sc.textFile(config.inputPath).zipWithIndex.toDF("line", "id")
    val df0 = df.filter(col("id") % 2 === 0).withColumn("y", col("line"))
    val df1 = df.filter(col("id") % 2 === 1).withColumn("x", col("line")).withColumn("id0", col("id") - 1)
    val af = df0.join(df1, df0.col("id") === df1.col("id0"))
    return af.select("x", "y")
  }
  
 def tokenModel(vocabSize: Int, labelSize: Int, config: Config): Sequential[Float] = {
    val model = Sequential()
    // input to an embedding layer is an index vector of `maxSeqquenceLength` elements, each index is in [0, vocabSize)
    // this layer produces a real-valued matrix of shape `maxSequenceLength x embeddingSize`
    model.add(Embedding(inputDim = vocabSize, outputDim = config.embeddingSize, inputLength=config.maxSequenceLength))
    // take the matrix above and feed to a GRU layer 
    // by default, the GRU layer produces a real-valued vector of length `recurrentSize` (the last output of the recurrent cell)
    // but since we want sequence information, we make it return a sequences, so the output will be a matrix of shape 
    // `maxSequenceLength x recurrentSize` 
    model.add(GRU(outputDim = config.recurrentSize, returnSequences = true))
    // feed the output of the GRU to a dense layer with relu activation function
    model.add(TimeDistributed(
      Dense(config.hiddenSize, activation="relu").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
      inputShape=Shape(config.maxSequenceLength, config.recurrentSize))
    )
    // add a dropout layer for regularization
    model.add(Dropout(config.dropoutProbability))
    // add the last layer for multi-class classification
    model.add(TimeDistributed(
      Dense(labelSize, activation="softmax").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
      inputShape=Shape(config.maxSequenceLength, config.hiddenSize))
    )
    return model
  }

  def tokenModelPreprocess(df: DataFrame, config: Config): (PipelineModel, Array[String], Array[String]) = {
    val xTokenizer = new Tokenizer().setInputCol("x").setOutputCol("xs")
    val xVectorizer = new CountVectorizer().setInputCol("xs").setOutputCol("us").setMinDF(config.minFrequency)
      .setVocabSize(config.vocabSize).setBinary(true)
    val yTokenizer = new Tokenizer().setInputCol("y").setOutputCol("ys")
    val yVectorizer = new CountVectorizer().setInputCol("ys").setOutputCol("vs").setBinary(true)
    val pipeline = new Pipeline().setStages(Array(xTokenizer, xVectorizer, yTokenizer, yVectorizer))
    val preprocessor = pipeline.fit(df)
    val vocabulary = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
    val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    println(s"vocabSize = ${vocabulary.size}, labels = ${labels.mkString}")
    return (preprocessor, vocabulary, labels)
  }

 def semiCharModel(vocabSize: Int, labelSize: Int, config: Config): Sequential[Float] = {
    val model = Sequential()
    // input to an embedding layer is an index vector of `3*maxSeqquenceLength` elements, each index is in [0, vocabSize)
    // this layer produces a real-valued matrix of shape `3*maxSequenceLength x embeddingSize`
    model.add(Embedding(inputDim = vocabSize, outputDim = config.embeddingSize, inputLength=3*config.maxSequenceLength))
    // reshape the matrix to matrix of shape `maxSequenceLength x 3*embeddingSize`. This operation perform the concatenation 
    // of [b, i, e] embedding vectors
    model.add(Reshape(targetShape=Array(-1, 3*config.embeddingSize)))
    // take the matrix above and feed to a GRU layer 
    // by default, the GRU layer produces a real-valued vector of length `recurrentSize` (the last output of the recurrent cell)
    // but since we want sequence information, we make it return a sequences, so the output will be a matrix of shape 
    // `maxSequenceLength x recurrentSize` 
    model.add(GRU(outputDim = config.recurrentSize, returnSequences = true))
    // feed the output of the GRU to a dense layer with relu activation function
    model.add(TimeDistributed(
      Dense(config.hiddenSize, activation="relu").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
      inputShape=Shape(config.maxSequenceLength, config.recurrentSize))
    )
    // add a dropout layer for regularization
    model.add(Dropout(config.dropoutProbability))
    // add the last layer for multi-class classification
    model.add(TimeDistributed(
      Dense(labelSize, activation="softmax").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
      inputShape=Shape(config.maxSequenceLength, config.hiddenSize))
    )
    return model
  }

  def semiCharPreprocess(df: DataFrame, config: Config): (PipelineModel, Array[String], Array[String]) = {
    val xTokenizer = new Tokenizer().setInputCol("x").setOutputCol("xs")
    val xTransformer = new SemiCharTransformer().setInputCol("xs").setOutputCol("ts")
    val xVectorizer = new CountVectorizer().setInputCol("ts").setOutputCol("us").setMinDF(config.minFrequency)
      .setVocabSize(config.vocabSize).setBinary(true)
    val yTokenizer = new Tokenizer().setInputCol("y").setOutputCol("ys")
    val yVectorizer = new CountVectorizer().setInputCol("ys").setOutputCol("vs").setBinary(true)
    val pipeline = new Pipeline().setStages(Array(xTokenizer, xTransformer, xVectorizer, yTokenizer, yVectorizer))
    val preprocessor = pipeline.fit(df)
    val vocabulary = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    val labels = preprocessor.stages(4).asInstanceOf[CountVectorizerModel].vocabulary
    println(s"vocabSize = ${vocabulary.size}, labels = ${labels.mkString}")
    return (preprocessor, vocabulary, labels)
  }

  def predict(df: DataFrame, config: Config) = {
    val inp = config.inputPath.split("/").last.split("""\.""").head
    val prefix = s"${config.modelPath}/${inp}/${config.modelType}"
    val preprocessor = PipelineModel.load(s"${prefix}/pre/")
    val model = Models.loadModel[Float](prefix + "/vsc.bigdl")
    // add a custom layer ArgMax as the last layer of this model so as to 
    // make the nnframes API of BigDL work. By default, the BigDL nnframes only process 2-d data (including the batch dimension)
    val sequential = model.asInstanceOf[Sequential[Float]]
    val argmax = ArgMaxLayer()
    sequential.add(argmax)

    val bf = preprocessor.transform(df)
    val cf = config.modelType match {
      case "tk" => 
        val vocabulary = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
        val vocabDict = vocabulary.zipWithIndex.toMap
        val xSequencer = new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
        xSequencer.transform(bf)
      case _    => 
        val vocabulary = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
        val vocabDict = vocabulary.zipWithIndex.toMap
        val xSequencer = new SemiCharSequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("ts").setOutputCol("features")
        xSequencer.transform(bf)
    }
    cf.show()
    // create a label map which idx -> label
    val labels = config.modelType match {
      case "tk" => preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
      case _ => preprocessor.stages(4).asInstanceOf[CountVectorizerModel].vocabulary
    }

    // transform the gold "ys" labels to indices
    val labelDict = labels.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    val ySequencer = new Sequencer(labelDict, config.maxSequenceLength, -1).setInputCol("ys").setOutputCol("label")    
    val ef = ySequencer.transform(cf)
    // run the prediction 
    val m = NNModel(sequential)
    val ff = m.transform(ef)
    val result = ff.select("prediction", "label")    

    // evaluate the result
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    val predictionsAndLabels = result.rdd.map { case row => 
      (row.getAs[Seq[Float]](0).toArray, row.getAs[Vector](1).toArray)
    }.flatMap { case (prediction, label) => 
      // truncate the padding values
      // find the first padding value (-1.0) in the label array
      var i = label.indexOf(-1.0)
      if (i == -1) i = label.size
      prediction.take(i).map(_.toDouble).zip(label.take(i).map(_.toDouble))
    }
    val metrics = new MulticlassMetrics(predictionsAndLabels)
    println(metrics.confusionMatrix)
    println(s"accuracy = ${metrics.accuracy}")
    // precision by label
    val ls = metrics.labels
    ls.foreach { l =>
      println(s"precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    ls.foreach { l =>
      println(s"recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    ls.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    ls.foreach { l =>
      println(s"fMeasure($l) = " + metrics.fMeasure(l))
    }
    println(labelDict)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val logger = LoggerFactory.getLogger(getClass.getName)

    val opts = new OptionParser[Config](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 8g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Int]('j', "layers").action((x, conf) => conf.copy(layers = x)).text("number of layers, default is 1")
      opt[Int]('r', "recurrentSize").action((x, conf) => conf.copy(recurrentSize = x)).text("number of hidden units in each recurrent layer")
      opt[Int]('h', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("number of hidden units in the dense layer")
      opt[Double]('n', "percentage").action((x, conf) => conf.copy(percentage = x)).text("percentage of the data set to use, default is 0.5")
      opt[Double]('u', "dropoutProbability").action((x, conf) => conf.copy(dropoutProbability = x)).text("dropout ratio, default is 0.2")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('l', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("max sequence length")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 0.001")
      opt[Boolean]('g', "gru").action((x, conf) => conf.copy(gru = x)).text("use 'gru' if true, otherwise use lstm")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('i', "inputPath").action((x, conf) => conf.copy(inputPath = x)).text("input data path")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode, default is false")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        implicit val formats = Serialization.formats(NoTypeHints)
        println(Serialization.writePretty(config))

        val conf = Engine.createSparkConf().setAppName(getClass().getName()).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        Engine.init

        val df = VSC.readData(sc, config).sample(config.percentage)
        df.printSchema()
        df.show()

        config.mode match {
          case "train" =>        
            val (pipelineModel, vocabulary, labels) = config.modelType match {
              case "tk" => VSC.tokenModelPreprocess(df, config)
              case _ => VSC.semiCharPreprocess(df, config)
            }
            // save the preprocessing pipeline for later loading
            val inp = config.inputPath.split("/").last.split("""\.""").head
            val prefix = s"${config.modelPath}/${inp}/${config.modelType}"
            pipelineModel.write.overwrite.save(s"${prefix}/pre/")
            

            val vocabDict = vocabulary.zipWithIndex.toMap
            val af = pipelineModel.transform(df)
            // map the label to one-based index (for use in ClassNLLCriterion of BigDL)
            val labelDict = labels.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
            val ySequencer = new Sequencer(labelDict, config.maxSequenceLength, -1).setInputCol("ys").setOutputCol("label")
            val bf = ySequencer.transform(af)

            val cf = config.modelType match {
              case "tk" => 
                val xSequencer = new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
                xSequencer.transform(bf)
              case _    => 
                val xSequencer = new SemiCharSequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("ts").setOutputCol("features")
                xSequencer.transform(bf)
            }
            cf.show()
            cf.printSchema()

            val Array(trainingDF, validationDF) = cf.randomSplit(Array(0.8, 0.2), seed = 80L)
          
            val model = config.modelType match {
              case "tk" => tokenModel(vocabulary.size, labels.size, config)
              case _    => semiCharModel(vocabulary.size, labels.size, config)
            }

            val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/${inp}/")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/${inp}/")
            val (featureSize, labelSize) = config.modelType match {
              case "tk" => (Array(config.maxSequenceLength), Array(config.maxSequenceLength))
              case "sc" => (Array(3*config.maxSequenceLength), Array(config.maxSequenceLength))
              case _    => (Array(0), Array(0))
            }
            val classifier = NNEstimator(model, TimeDistributedCriterion(ClassNLLCriterion(), true), featureSize, labelSize)
                .setLabelCol("label").setFeaturesCol("features")
                .setBatchSize(config.batchSize)
                .setOptimMethod(new Adam(config.learningRate))
                .setMaxEpoch(config.epochs)
                .setTrainSummary(trainingSummary)
                .setValidationSummary(validationSummary)
                .setValidation(Trigger.everyEpoch, validationDF, Array(new TimeDistributedTop1Accuracy(paddingValue = -1)), config.batchSize)
            classifier.fit(trainingDF)

            val trainingAccuracy = trainingSummary.readScalar("TimeDistributedTop1Accuracy")
            val validationLoss = validationSummary.readScalar("Loss")
            val validationAccuracy = validationSummary.readScalar("TimeDistributedTop1Accuracy")
            logger.info("     Train Accuracy: " + trainingAccuracy.mkString(", "))
            logger.info("Validation Accuracy: " + validationAccuracy.mkString(", "))

            logger.info("Saving the model...")        
            model.saveModel(prefix + "/vsc.bigdl")
        case "predict" => 
          predict(df, config)
      }
        sc.stop()
      case None => {}
    }
  }
}
