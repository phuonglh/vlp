package vlp.woz.act

import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.keras.models.{Models, KerasNet}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}

import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.CountVectorizerModel

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

import java.nio.file.{Files, Paths, StandardOpenOption}

import vlp.woz.DialogReader

/**
  * phuonglh@gmail.com
  * 
  * February 2023
  * 
  */
object ClassifierFPT {
  implicit val formats = Serialization.formats(NoTypeHints)

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[Config](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 8g")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language, either en or vi")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('e', "embeddingSize").action((x, conf) => conf.copy(embeddingSize = x)).text("embedding size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
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
        spark.sparkContext.setLogLevel("ERROR")

        // create a model
        val model = ModelFactory(config)
        val prefix = s"${config.modelPath}/vi/${config.modelType}"
        val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/act/vi/")
        val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/act/vi/")
        // read all the act
        val df = spark.read.json("dat/vie/act")
        val Array(trainingDF, validationDF, testDF) = df.randomSplit(Array(0.8, 0.1, 0.1), 220712L)
        testDF.show()

        config.mode match {
          case "train" => 
            println(Serialization.writePretty(config))
            val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
            val bigdl = Classifier.train(model, config, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
            // save the model
            preprocessor.write.overwrite.save(s"${config.modelPath}/vi/pre/")
            bigdl.saveModel(prefix + "/act.bigdl", overWrite = true)
            val trainingAccuracy = trainingSummary.readScalar("Top1Accuracy")
            val validationLoss = validationSummary.readScalar("Loss")
            val validationAccuracy = validationSummary.readScalar("Top1Accuracy")
            println("Train Accuracy: " + trainingAccuracy.mkString(", "))
            println("Valid Accuracy: " + validationAccuracy.mkString(", "))
            println("Validation Loss: " + validationLoss.mkString(", "))
            // transform actNames to a sequence of labels
            val labelDict = labels.zipWithIndex.toMap
            println(labelDict.toString)
            val labelIndexer = new SequenceIndexer(labelDict).setInputCol("actNames").setOutputCol("target")
            val (cft, cfv) = (labelIndexer.transform(trainingDF), labelIndexer.transform(validationDF))
            // training score
            val dft = model.predict(cft, preprocessor, bigdl)
            val trainingScore = Classifier.evaluate(dft, labels.size, config, "train")
            Classifier.saveScore(trainingScore, config.scorePath)
            // validation score
            val dfv = model.predict(cfv, preprocessor, bigdl)
            val validationScore = Classifier.evaluate(dfv, labels.size, config, "valid")
            Classifier.saveScore(validationScore, config.scorePath)
            // test score
            val xf = labelIndexer.transform(testDF)
            val test = model.predict(xf, preprocessor, bigdl)
            val testScore = Classifier.evaluate(test, labels.size, config, "test")
            Classifier.saveScore(testScore, config.scorePath)
          case "eval" => 
            println(s"Loading preprocessor ${config.modelPath}/vi/pre/...")
            val preprocessor = PipelineModel.load(s"${config.modelPath}/vi/pre/")
            println(s"Loading model ${prefix}/act.bigdl...")
            var bigdl = Models.loadModel[Float](prefix + "/act.bigdl")
            // transform actNames to a sequence of labels
            val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
            val labelDict = labels.zipWithIndex.toMap
            println(labelDict.toString)
            val labelIndexer = new SequenceIndexer(labelDict).setInputCol("actNames").setOutputCol("target")
            val dft = labelIndexer.transform(trainingDF)
            val model = ModelFactory(config)
            // training score
            val trainingResult = model.predict(dft, preprocessor, bigdl)
            trainingResult.show(false)
            var score = Classifier.evaluate(trainingResult, labels.size, config, "train")
            Classifier.saveScore(score, config.scorePath)
            // validation score
            val dfv = labelIndexer.transform(validationDF)
            val validationResult = model.predict(dfv, preprocessor, bigdl)
            score = Classifier.evaluate(validationResult, labels.size, config, "valid")
            Classifier.saveScore(score, config.scorePath)
            // test score
            val xf = labelIndexer.transform(testDF)
            val test = model.predict(xf, preprocessor, bigdl)
            score = Classifier.evaluate(test, labels.size, config, "test")
            Classifier.saveScore(score, config.scorePath)

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
                val bigdl = Classifier.train(model, conf, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
                // training score
                val dft = model.predict(cft, preprocessor, bigdl)
                val trainingScore = Classifier.evaluate(dft, labels.size, conf, "train")
                Classifier.saveScore(trainingScore, config.scorePath)
                // validation score
                val dfv = model.predict(cfv, preprocessor, bigdl)
                val validationScore = Classifier.evaluate(dfv, labels.size, conf, "valid")
                Classifier.saveScore(validationScore, config.scorePath)
                // test score                
                val test = model.predict(xf, preprocessor, bigdl)
                val testScore = Classifier.evaluate(test, labels.size, conf, "test")
                Classifier.saveScore(testScore, config.scorePath)
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
                val bigdl = Classifier.train(model, conf, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
                // training score
                val dft = model.predict(cft, preprocessor, bigdl)
                val trainingScore = Classifier.evaluate(dft, labels.size, conf, "train")
                Classifier.saveScore(trainingScore, config.scorePath)
                // validation score
                val dfv = model.predict(cfv, preprocessor, bigdl)
                val validationScore = Classifier.evaluate(dfv, labels.size, conf, "valid")
                Classifier.saveScore(validationScore, config.scorePath)
                // test score                
                val test = model.predict(xf, preprocessor, bigdl)
                val testScore = Classifier.evaluate(test, labels.size, conf, "test")
                Classifier.saveScore(testScore, config.scorePath)
              }
            }
          case _ => println("What mode do you want to run?")
        }
        sc.stop()
      case None => {}
    }
  }    
}
