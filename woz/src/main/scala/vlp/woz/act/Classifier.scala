package vlp.woz.act

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.keras.models.{Models, KerasNet}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.keras.metrics.{CategoricalAccuracy, Top5Accuracy}
import com.intel.analytics.bigdl.dllib.optim.{MeanAveragePrecision}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.optim.{Loss, MAE, Trigger}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.nnframes.{NNModel, NNEstimator}
import com.intel.analytics.bigdl.dllib.nn.{TimeDistributedCriterion, BCECriterion, MSECriterion}

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{CountVectorizerModel, VectorAssembler, RegexTokenizer}

import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession, DataFrame}

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

import org.apache.log4j.{Logger, Level}
import org.slf4j.LoggerFactory
import java.nio.file.{Files, Paths, StandardOpenOption}
import org.apache.spark.mllib.evaluation.MultilabelMetrics

import vlp.woz.DialogReader

/**
  * phuonglh@gmail.com
  * 
  * February 2023
  * 
  */
object Classifier {
  implicit val formats = Serialization.formats(NoTypeHints)

  def train(model: AbstractModel, config: Config, trainingDF: DataFrame, validationDF: DataFrame, 
    preprocessor: PipelineModel, vocabulary: Array[String], labels: Array[String], 
    trainingSummary: TrainSummary, validationSummary: ValidationSummary): KerasNet[Float] = {
    val bigdl = model.createModel(vocabulary.size, labels.size)
    bigdl.summary()
    // build a vocab map
    val vocabDict = vocabulary.zipWithIndex.toMap

    val (aft, afv) = (preprocessor.transform(trainingDF), preprocessor.transform(validationDF))

    val (cft, cfv) = if (config.modelType == "lstm") {
      val xSequencer = new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
      (xSequencer.transform(aft), xSequencer.transform(afv))
    } else if (config.modelType == "bert") { 
      val xSequencer = new Sequencer4BERT(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features") 
      (xSequencer.transform(aft), xSequencer.transform(afv))
    } else { // lstm-boa
      val xSequencer = new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("ts")
      val (cft1, cfv1) = (xSequencer.transform(aft), xSequencer.transform(afv))
      val assembler = new VectorAssembler().setInputCols(Array("ts", "ys")).setOutputCol("features")
      (assembler.transform(cft1), assembler.transform(cfv1))
    }
    cfv.printSchema()

    val maxSeqLen = config.maxSequenceLength
    val estimator = if (config.modelType == "lstm") {
      val featureSize = Array(maxSeqLen)
      val labelSize = Array(labels.size)
      NNEstimator(bigdl, BCECriterion(), featureSize, labelSize)
    } else if (config.modelType == "bert") {
      val featureSize = Array(Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen))
      val labelSize = Array(labels.size)
      NNEstimator(bigdl, BCECriterion(), featureSize, labelSize)
    } else { // lstm-boa
      val featureSize = Array(Array(maxSeqLen), Array(labels.size))
      val labelSize = Array(labels.size)
      NNEstimator(bigdl, BCECriterion(), featureSize, labelSize)
    }
    estimator.setLabelCol("label").setFeaturesCol("features")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(config.learningRate))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, cfv, Array(new CategoricalAccuracy(), new MAE()), config.batchSize)
    estimator.fit(cft)
    return bigdl
  }

  def evaluate(result: DataFrame, labelSize: Int, config: Config, split: String): Score = {
    // evaluate the result
    val predictionsAndLabels = result.rdd.map { case row => 
      (row.getAs[Seq[Double]](0).toArray, row.getAs[Seq[Double]](1).toArray)
    }
    val metrics = new MultilabelMetrics(predictionsAndLabels)
    val precisionByLabel = Array.fill(labelSize)(0d)
    val recallByLabel = Array.fill(labelSize)(0d)
    val fMeasureByLabel = Array.fill(labelSize)(0d)
    val ls = metrics.labels
    println(ls.mkString(", "))
    ls.foreach { k => 
      precisionByLabel(k.toInt) = metrics.precision(k)
      recallByLabel(k.toInt) = metrics.recall(k)
      fMeasureByLabel(k.toInt) = metrics.f1Measure(k)
    }
    Score(
      config.language,
      config.modelType, split,
      if (config.modelType.startsWith("lstm")) config.embeddingSize else -1,
      if (config.modelType == "bert") config.bert.hiddenSize else config.recurrentSize,
      if (config.modelType == "bert") config.bert.nBlock else config.layers,
      if (config.modelType == "bert") config.bert.nHead else -1,
      if (config.modelType == "bert") config.bert.intermediateSize else -1,
      metrics.accuracy, metrics.f1Measure, metrics.microF1Measure, metrics.microPrecision, metrics.microRecall,
      precisionByLabel, recallByLabel, fMeasureByLabel
    )
  }

  def saveScore(score: Score, path: String) = {
    var content = Serialization.writePretty(score) + ",\n"
    Files.write(Paths.get(path), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
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
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language, either en or vi")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Int]('e', "embeddingSize").action((x, conf) => conf.copy(embeddingSize = x)).text("embedding size")
      opt[Int]('j', "layers").action((x, conf) => conf.copy(layers = x)).text("number of layers, default is 1")
      opt[Int]('r', "recurrentSize").action((x, conf) => conf.copy(recurrentSize = x)).text("number of hidden units in each recurrent layer")
      opt[Double]('u', "dropoutProbability").action((x, conf) => conf.copy(dropoutProbability = x)).text("dropout ratio")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 5E-4")
      opt[String]('d', "trainPath").action((x, conf) => conf.copy(trainPath = x)).text("training data directory")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[String]('s', "scorePath").action((x, conf) => conf.copy(scorePath = x)).text("score path")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode, default is false")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val conf = Engine.createSparkConf().setAppName(getClass().getName()).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        Engine.init
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()

        // create a model
        val model = ModelFactory(config)
        val prefix = s"${config.modelPath}/${config.language}/${config.modelType}"
        val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/act/${config.language}")
        val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/act/${config.language}")
        // read train/dev datasets
        val (trainingDF0, validationDF0) = (spark.read.json(config.trainPath), spark.read.json(config.devPath))
        val testDF0 = spark.read.json(config.testPath)

        val trainingDF1 = DialogReader.concatDialogActs(spark, trainingDF0)
        val validationDF1 = DialogReader.concatDialogActs(spark, validationDF0)
        val testDF1 = DialogReader.concatDialogActs(spark, testDF0)
        val prevTokenizer = new RegexTokenizer().setInputCol("prevActs").setOutputCol("ps").setPattern("""[\s]+""")
        val (trainingDF, validationDF) = (prevTokenizer.transform(trainingDF1), prevTokenizer.transform(validationDF1))
        val testDF = prevTokenizer.transform(testDF1)
        testDF.show()

        config.mode match {
          case "train" => 
            logger.info(Serialization.writePretty(config))
            val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
            val bigdl = train(model, config, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
            // save the model
            preprocessor.write.overwrite.save(s"${prefix}/pre/")
            logger.info("Saving the model...")        
            bigdl.saveModel(prefix + "/act.bigdl", overWrite = true)
            val trainingAccuracy = trainingSummary.readScalar("Top1Accuracy")
            val validationLoss = validationSummary.readScalar("Loss")
            val validationAccuracy = validationSummary.readScalar("Top1Accuracy")
            logger.info("Train Accuracy: " + trainingAccuracy.mkString(", "))
            logger.info("Valid Accuracy: " + validationAccuracy.mkString(", "))
            logger.info("Validation Loss: " + validationLoss.mkString(", "))
            // transform actNames to a sequence of labels
            val labelDict = labels.zipWithIndex.toMap
            logger.info(labelDict.toString)
            val labelIndexer = new SequenceIndexer(labelDict).setInputCol("actNames").setOutputCol("target")
            val (cft, cfv) = (labelIndexer.transform(trainingDF), labelIndexer.transform(validationDF))
            // training score
            val dft = model.predict(cft, preprocessor, bigdl)
            val trainingScore = evaluate(dft, labels.size, config, "train")
            logger.info(s"${Serialization.writePretty(trainingScore)}") 
            saveScore(trainingScore, config.scorePath)
            // validation score
            val dfv = model.predict(cfv, preprocessor, bigdl)
            val validationScore = evaluate(dfv, labels.size, config, "valid")
            logger.info(s"${Serialization.writePretty(validationScore)}") 
            saveScore(validationScore, config.scorePath)
            // test score
            val xf = labelIndexer.transform(testDF)
            val test = model.predict(xf, preprocessor, bigdl)
            val testScore = evaluate(test, labels.size, config, "test")
            logger.info(s"${Serialization.writePretty(testScore)}") 
            saveScore(testScore, config.scorePath)
          case "eval" => 
            logger.info(s"Loading preprocessor ${prefix}/pre/...")
            val preprocessor = PipelineModel.load(s"${prefix}/pre/")
            logger.info(s"Loading model ${prefix}/act.bigdl...")
            var bigdl = Models.loadModel[Float](prefix + "/act.bigdl")
            // transform actNames to a sequence of labels
            val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
            val labelDict = labels.zipWithIndex.toMap
            logger.info(labelDict.toString)
            val labelIndexer = new SequenceIndexer(labelDict).setInputCol("actNames").setOutputCol("target")
            val dft = labelIndexer.transform(trainingDF)
            val model = ModelFactory(config)
            // training score
            val trainingResult = model.predict(dft, preprocessor, bigdl)
            trainingResult.show(false)
            var score = evaluate(trainingResult, labels.size, config, "train")
            logger.info(s"${Serialization.writePretty(score)}") 
            saveScore(score, config.scorePath)
            // validation score
            val dfv = labelIndexer.transform(validationDF)
            val validationResult = model.predict(dfv, preprocessor, bigdl)
            score = evaluate(validationResult, labels.size, config, "valid")
            logger.info(s"${Serialization.writePretty(score)}") 
            saveScore(score, config.scorePath)
            // test score
            val xf = labelIndexer.transform(testDF)
            val test = model.predict(xf, preprocessor, bigdl)
            score = evaluate(test, labels.size, config, "test")
            logger.info(s"${Serialization.writePretty(score)}") 
            saveScore(score, config.scorePath)

          case "experiment-lstm" => 
            // Perform multiple experiments with token LSTM model. There are 45 configurations, each is run 3 times.
            val embeddingSizes = Seq(16, 32, 64)
            val recurrentSizes = Seq(32, 64, 128, 256, 512)
            val layerSizes = Seq(1, 2, 3)
            val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
            val labelDict = labels.zipWithIndex.toMap
            val labelIndexer = new SequenceIndexer(labelDict).setInputCol("actNames").setOutputCol("target")
            val (cft, cfv) = (labelIndexer.transform(trainingDF), labelIndexer.transform(validationDF))
            val xf = labelIndexer.transform(testDF)
            val t = config.modelType // [lstm, lstm-boa]
            for (e <- embeddingSizes; r <- recurrentSizes; j <- layerSizes) {
              // note that the model type is passed by the global configuration through the command line
              val conf = Config(modelType = t, embeddingSize = e, recurrentSize = r, layers = j, batchSize = config.batchSize)
              val model = ModelFactory(conf)
              // each config will be run 3 times
              for (k <- 0 to 2) {
                val bigdl = train(model, conf, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
                // training score
                val dft = model.predict(cft, preprocessor, bigdl)
                val trainingScore = evaluate(dft, labels.size, conf, "train")
                saveScore(trainingScore, config.scorePath)
                // validation score
                val dfv = model.predict(cfv, preprocessor, bigdl)
                val validationScore = evaluate(dfv, labels.size, conf, "valid")
                saveScore(validationScore, config.scorePath)
                // test score                
                val test = model.predict(xf, preprocessor, bigdl)
                val testScore = evaluate(test, labels.size, conf, "test")
                saveScore(testScore, config.scorePath)
              }
            }
          case "experiment-bert" =>
            // Perform multiple experiments with token BERT model. There are 54 configurations, each is run 3 times.
            val hiddenSizes = Seq(16, 32, 64)
            val nBlocks = Seq(2, 4, 8)
            val nHeads = Seq(2, 4, 8)
            val intermediateSizes = Seq(32, 64)
            val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
            val labelDict = labels.zipWithIndex.toMap
            val labelIndexer = new SequenceIndexer(labelDict).setInputCol("actNames").setOutputCol("target")
            val (cft, cfv) = (labelIndexer.transform(trainingDF), labelIndexer.transform(validationDF))
            val xf = labelIndexer.transform(testDF)
            for (hiddenSize <- hiddenSizes; nBlock <- nBlocks; nHead <- nHeads; intermediateSize <- intermediateSizes) {
              val bertConfig = ConfigBERT(hiddenSize, nBlock, nHead, config.maxSequenceLength, intermediateSize)
              val conf = Config(modelType = "bert", bert = bertConfig, batchSize = config.batchSize)
              val model = ModelFactory(conf)
              // each config will be run 3 times
              for (k <- 0 to 2) {
                val bigdl = train(model, conf, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
                // training score
                val dft = model.predict(cft, preprocessor, bigdl)
                val trainingScore = evaluate(dft, labels.size, conf, "train")
                saveScore(trainingScore, config.scorePath)
                // validation score
                val dfv = model.predict(cfv, preprocessor, bigdl)
                val validationScore = evaluate(dfv, labels.size, conf, "valid")
                saveScore(validationScore, config.scorePath)
                // test score                
                val test = model.predict(xf, preprocessor, bigdl)
                val testScore = evaluate(test, labels.size, conf, "test")
                saveScore(testScore, config.scorePath)
              }
            }
          case _ => logger.error("What mode do you want to run?")
        }
        sc.stop()
      case None => {}
    }
  }    
}
