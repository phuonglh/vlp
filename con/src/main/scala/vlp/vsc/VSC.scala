package vlp.vsc

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.keras.models.{Models, KerasNet}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.optim.{Loss, Trigger}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.nnframes.{NNModel, NNEstimator}
import com.intel.analytics.bigdl.dllib.nn.{TimeDistributedCriterion, ClassNLLCriterion}

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.linalg.Vector

import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession, DataFrame}

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

import org.apache.log4j.{Logger, Level}
import org.slf4j.LoggerFactory
import java.nio.file.{Files, Paths, StandardOpenOption}
import scala.concurrent.{Future, Await}
import scala.concurrent.duration._

import vlp.con.{Sequencer, CharSequencer, Sequencer4BERT, TimeDistributedTop1Accuracy}


object VSC {
  implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global
  
  def dataPaths(language: String): (String, String) = {
    language match {
      case "czech" => ("dat/ged/czech/cs_geccc_train.tsv", "dat/ged/czech/cs_geccc_dev.tsv")
      case "english" => ("dat/ged/english/en_fce_train.tsv", "dat/ged/english/en_fce_dev.tsv")
      case "german" => ("dat/ged/german/de_falko-merlin_train.tsv", "dat/ged/german/de_falko-merlin_dev.tsv")
      case "italian" => ("dat/ged/italian/it_merlin_train.tsv", "dat/ged/italian/it_merlin_dev.tsv")
      case "swedish" => ("dat/ged/swedish/sv_swell_train.tsv", "dat/ged/swedish/sv_swell_dev.tsv")
      case _ => ("dat/vsc/100.txt.inp", "dat/vsc/100.txt.inp")
    }
  }

  def testPath(language: String): String = {
    language match {
      case "czech" => "dat/ged/czech/cs_geccc_test_unlabelled.tsv"
      case "english" => "dat/ged/english/en_fce_test_unlabelled.tsv"
      case "german" => "dat/ged/german/de_falko-merlin_test_unlabelled.tsv"
      case "italian" => "dat/ged/italian/it_merlin_test_unlabelled.tsv"
      case "swedish" => "dat/ged/swedish/sv_swell_test_unlabelled.tsv"
      case _ => ""
    }
  }

  def evaluate(result: DataFrame, labelSize: Int, config: Config, split: String): Score = {
    // evaluate the result
    val predictionsAndLabels = result.rdd.map{ case row => 
      (row.getAs[Seq[Float]](0).toArray, row.getAs[Vector](1).toArray)
    }.flatMap { case (prediction, label) => 
      // truncate the padding values
      // find the first padding value (-1.0) in the label array
      var i = label.indexOf(-1.0)
      if (i == -1) i = label.size
      prediction.take(i).map(_.toDouble).zip(label.take(i).map(_.toDouble))
    }
    val metrics = new org.apache.spark.mllib.evaluation.MulticlassMetrics(predictionsAndLabels)
    val precisionByLabel = Array.fill(labelSize)(0d)
    val recallByLabel = Array.fill(labelSize)(0d)
    val fMeasureByLabel = Array.fill(labelSize)(0d)
    // precision by label: our BigDL model uses 1-based labels, so we need to decrease 1 unit.
    val ls = metrics.labels
    println(ls.mkString(", "))
    ls.foreach { k => 
      precisionByLabel(k.toInt-1) = metrics.precision(k)
      recallByLabel(k.toInt-1) = metrics.recall(k)
      fMeasureByLabel(k.toInt-1) = metrics.fMeasure(k, 0.5) // beta = 0.5
    }
    val inp = if (config.ged) config.language else config.inputPath.split("/").last.split("""\.""").head
    Score(
      inp, config.modelType, split,
      if (Seq("tk", "st").contains(config.modelType)) config.embeddingSize else -1,
      if (Seq("tb", "sb").contains(config.modelType)) config.bert.hiddenSize else config.recurrentSize,
      if (Seq("tb", "sb").contains(config.modelType)) config.bert.nBlock else config.layers,
      if (Seq("tb", "sb").contains(config.modelType)) config.bert.nHead else -1,
      if (Seq("tb", "sb").contains(config.modelType)) config.bert.intermediateSize else -1,
      metrics.confusionMatrix, metrics.accuracy, precisionByLabel, recallByLabel, fMeasureByLabel
    )
  }

  def train(model: AbstractModel, config: Config, trainingDF: DataFrame, validationDF: DataFrame, 
    preprocessor: PipelineModel, vocabulary: Array[String], labels: Array[String], 
    trainingSummary: TrainSummary, validationSummary: ValidationSummary): KerasNet[Float] = {
    val bigdl = model.createModel(vocabulary.size, labels.size)
    bigdl.summary()
    // build a vocab map
    val vocabDict = vocabulary.zipWithIndex.toMap
    // map the label to one-based index (for use in ClassNLLCriterion of BigDL), label padding value is -1.
    val labelDict = labels.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    val ySequencer = new Sequencer(labelDict, config.maxSequenceLength, -1).setInputCol("ys").setOutputCol("label")

    val xSequencer = config.modelType match {
      case "tk" => new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
      case "tb" => new Sequencer4BERT(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
      case "st" => new SubtokenSequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("ts").setOutputCol("features")
      case _ => new CharSequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
    }

    // make sure the dataframes are ready before launching a training process
    val future: Future[(DataFrame, DataFrame)] = Future {
      val (aft, afv) = (preprocessor.transform(trainingDF), preprocessor.transform(validationDF))
      val (bft, bfv) = (ySequencer.transform(aft), ySequencer.transform(afv))
      (xSequencer.transform(bft), xSequencer.transform(bfv))
    }
    val (cft, cfv) = Await.result(future, Duration.Inf)
    cfv.printSchema()

    // our classes are unbalanced, hence we use weights to improve accuracy
    // the first label is more common than all the rest, hence it takes a lesser weight
    val w = Tensor(Array(labelDict.size)).rand()
    w.setValue(1, 0.1f); for (j <- 2 to labelDict.size) w.setValue(j, 0.9f)

    val maxSeqLen = config.maxSequenceLength
    val classifier = config.modelType match {
      case "tk" => 
        val (featureSize, labelSize) = (Array(maxSeqLen), Array(maxSeqLen))
        NNEstimator(bigdl, TimeDistributedCriterion(ClassNLLCriterion(weights=w, logProbAsInput=false), true), featureSize, labelSize)
      case "tb" => 
        val (featureSize, labelSize) = (Array(Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen)), Array(maxSeqLen))
        NNEstimator(bigdl, TimeDistributedCriterion(ClassNLLCriterion(weights=w, logProbAsInput=false), true), featureSize, labelSize)
      case "st" => 
        val (featureSize, labelSize) = (Array(3*maxSeqLen), Array(maxSeqLen))
        NNEstimator(bigdl, TimeDistributedCriterion(ClassNLLCriterion(weights=w, logProbAsInput=false), true), featureSize, labelSize)
      case "ch" => 
        val (featureSize, labelSize) = (Array(3*maxSeqLen*vocabulary.size), Array(maxSeqLen))
        NNEstimator(bigdl, TimeDistributedCriterion(ClassNLLCriterion(weights=w, logProbAsInput=false), true), featureSize, labelSize)
      case _ => 
        val (featureSize, labelSize) = (Array(0), Array(0))
        NNEstimator(bigdl, TimeDistributedCriterion(ClassNLLCriterion(weights=w, logProbAsInput=false), true), featureSize, labelSize)
    }

    classifier.setLabelCol("label").setFeaturesCol("features")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(config.learningRate))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, cfv, Array(new TimeDistributedTop1Accuracy(paddingValue = -1)), config.batchSize)
    // fit the classifier, which will train the bigdl model and return a NNModel
    // but we cannot use this NNModel to transform because we need a custom layer ArgMaxLayer 
    // at the end to output a good format for BigDL. See the predict() method for detail.
    classifier.fit(cft)
    return bigdl
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
      opt[Double]('n', "percentage").action((x, conf) => conf.copy(percentage = x)).text("percentage of the data set to use")
      opt[Double]('u', "dropoutProbability").action((x, conf) => conf.copy(dropoutProbability = x)).text("dropout ratio")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language {czech, english, german, italian, swedish, vietnamese}")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 0.001")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('i', "inputPath").action((x, conf) => conf.copy(inputPath = x)).text("input data path")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[String]('s', "scorePath").action((x, conf) => conf.copy(scorePath = x)).text("score path")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode, default is false")
      opt[Unit]('g', "GED").action((_, conf) => conf.copy(ged = true)).text("GED mode, default is false")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        implicit val formats = Serialization.formats(NoTypeHints)

        val conf = Engine.createSparkConf().setAppName(getClass().getName()).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        Engine.init
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()

        val (trainPath, validPath) = dataPaths(config.language)
        val Array(trainingDF, validationDF) = if (config.ged) {
          // separate train/dev split
          Array(DataReader.readDataGED(sc, trainPath), DataReader.readDataGED(sc, validPath))
        } else {
          // personal data set with 80/20 of train/valid split
          val df = DataReader.readData(sc, config.inputPath).sample(config.percentage)
          df.randomSplit(Array(0.8, 0.2), seed = 85L)
        }
        trainingDF.printSchema()
        trainingDF.show()
        // create a model
        val model = ModelFactory(config)
        // get the input data set name (for example, "vud", "fin", "english", "italian") and create a prefix
        val inp = if (config.ged) config.language else config.inputPath.split("/").last.split("""\.""").head
        val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/${inp}/")
        val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/${inp}/")

        config.mode match {
          case "train" => 
            logger.info(Serialization.writePretty(config))
            val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
            val bigdl = train(model, config, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
            // save the model
            val prefix = s"${config.modelPath}/${inp}/${config.modelType}"
            preprocessor.write.overwrite.save(s"${prefix}/pre/")
            logger.info("Saving the model...")        
            bigdl.saveModel(prefix + "/vsc.bigdl", overWrite = true)

            val trainingAccuracy = trainingSummary.readScalar("TimeDistributedTop1Accuracy")
            val validationAccuracy = validationSummary.readScalar("TimeDistributedTop1Accuracy")
            logger.info("Train Accuracy: " + trainingAccuracy.mkString(", "))
            logger.info("Valid Accuracy: " + validationAccuracy.mkString(", "))
            // evaluate on the training data
            val dft = model.predict(trainingDF, preprocessor, bigdl, true)
            val trainingScores = evaluate(dft, labels.size, config, "train")
            logger.info(s"Training score: ${Serialization.writePretty(trainingScores)}") 
            var content = Serialization.writePretty(trainingScores) + ",\n"
            Files.write(Paths.get(config.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
            // evaluate on the validation data (don't add the second ArgMaxLayer at the end)
            val dfv = model.predict(validationDF, preprocessor, bigdl, false)
            val validationScores = evaluate(dfv, labels.size, config, "valid")
            logger.info(s"Validation score: ${Serialization.writePretty(validationScores)}")
            content = Serialization.writePretty(validationScores) + ",\n"
            Files.write(Paths.get(config.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
        case "eval" => 
          val prefix = s"${config.modelPath}/${inp}/${config.modelType}"
          logger.info(s"Loading preprocessor ${prefix}/pre/...")
          val preprocessor = PipelineModel.load(s"${prefix}/pre/")
          logger.info(s"Loading model ${prefix}/vsc.bigdl...")
          var bigdl = Models.loadModel[Float](prefix + "/vsc.bigdl")
          val labels = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
          val labelDict = labels.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
          val model = ModelFactory(config)
          val trainingResult = model.predict(trainingDF, preprocessor, bigdl, true)
          var scores = evaluate(trainingResult, labels.size, config, "train")
          logger.info(Serialization.writePretty(scores))
          var content = Serialization.writePretty(scores) + ",\n"
          Files.write(Paths.get(config.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
          val result = model.predict(validationDF, preprocessor, bigdl, false)
          scores = evaluate(result, labels.size, config, "valid")
          logger.info(Serialization.writePretty(scores))
          content = Serialization.writePretty(scores) + ",\n"
          Files.write(Paths.get(config.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
        case "experiment-tk-st" => 
          // perform multiple experiments for a given language
          // Multi-layer LSTM models and two representations (token, subtoken) are run and compared.
          // Different hyper-parameters are tried: 27 triples of hyper-parameter configs, each is run 3 times.
          // So, there are 81 runs for each language in this experiment.
          val embeddingSizes = Seq(16, 32, 64)
          val recurrentSizes = Seq(32, 64, 128)
          val layerSizes = Seq(1, 2, 3)
          // need to wait the preprocessor thread to finish before launching experiment: JOKING, not necessary!
          val future: Future[(PipelineModel, Array[String], Array[String])] = Future {
            model.preprocessor(trainingDF)
          }
          val (preprocessor, vocabulary, labels) = Await.result(future, Duration.Inf)
          // val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
          val scorePath = s"dat/vsc/scores-tk-st-${config.language}.json"
          for (e <- embeddingSizes; r <- recurrentSizes; j <- layerSizes) {
            // note that the model type is passed by the global configuration through the command line
            val conf = Config(modelType = config.modelType, embeddingSize = e, recurrentSize = r, layers = j, language = config.language, ged = config.ged, batchSize = config.batchSize)
            logger.info(Serialization.writePretty(conf))
            val model = ModelFactory(conf)
            // each config will be run 3 times
            for (k <- 0 to 2) {
              val bigdl = train(model, conf, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
              // evaluate on the training data
              val dft = model.predict(trainingDF, preprocessor, bigdl, true)
              val trainingScores = evaluate(dft, labels.size, conf, "train")
              var content = Serialization.writePretty(trainingScores) + ",\n"
              Files.write(Paths.get(scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
              // evaluate on the validation data (don't add the second ArgMaxLayer at the end)
              val dfv = model.predict(validationDF, preprocessor, bigdl, false)
              val validationScores = evaluate(dfv, labels.size, conf, "valid")
              content = Serialization.writePretty(validationScores) + ",\n"
              Files.write(Paths.get(scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
            }       
          }
        case "experiment-ch" => 
          // perform multiple experiments for a given language with the character-based model.
          // Different hyper-parameters are tried: 9 triples of hyper-parameter configs, each is run 3 times.
          // So, there are 27 runs for each language in this experiment.
          val recurrentSizes = Seq(32, 64, 128)
          val layerSizes = Seq(1, 2, 3)
          val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
          val scorePath = s"dat/vsc/scores-ch-${config.language}.json"
          for (r <- recurrentSizes; j <- layerSizes) {
            val conf = Config(modelType = "ch", recurrentSize = r, layers = j, language = config.language, ged = config.ged, batchSize = config.batchSize,
              driverMemory = config.driverMemory, executorMemory = config.executorMemory)
            logger.info(Serialization.writePretty(conf))
            val model = ModelFactory(conf)
            // each config will be run 3 times
            for (k <- 0 to 2) {
              val bigdl = train(model, conf, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
              // evaluate on the training data
              val dft = model.predict(trainingDF, preprocessor, bigdl, true)
              val trainingScores = evaluate(dft, labels.size, conf, "train")
              var content = Serialization.writePretty(trainingScores) + ",\n"
              Files.write(Paths.get(scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
              // evaluate on the validation data (don't add the second ArgMaxLayer at the end)
              val dfv = model.predict(validationDF, preprocessor, bigdl, false)
              val validationScores = evaluate(dfv, labels.size, conf, "valid")
              content = Serialization.writePretty(validationScores) + ",\n"
              Files.write(Paths.get(scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
            }       
          }
        case "experiment-tb" =>
          // perform multiple experiments with token BERT model. 
          // There are 54 configurations, each is run 5 times.
          val hiddenSizes = Seq(16, 32, 64)
          val nBlocks = Seq(2, 4, 8)
          val nHeads = Seq(2, 4, 8)
          val intermediateSizes = Seq(32, 64)
          val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
          val scorePath = s"dat/vsc/scores-tb-${config.language}.json"
          for (hiddenSize <- hiddenSizes; nBlock <- nBlocks; nHead <- nHeads; intermediateSize <- intermediateSizes) {
            val bertConfig = ConfigBERT(hiddenSize, nBlock, nHead, config.maxSequenceLength, intermediateSize)
            val conf = Config(modelType = "tb", language = config.language, ged = config.ged, bert = bertConfig, batchSize = config.batchSize,
              driverMemory = config.driverMemory, executorMemory = config.executorMemory)
            logger.info(Serialization.writePretty(conf))
            val model = ModelFactory(conf)
            // each config will be run 5 times
            for (k <- 0 to 4) {
              val bigdl = train(model, conf, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
              // evaluate on the training data
              val dft = model.predict(trainingDF, preprocessor, bigdl, true)
              val trainingScores = evaluate(dft, labels.size, conf, "train")
              var content = Serialization.writePretty(trainingScores) + ",\n"
              Files.write(Paths.get(scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
              // evaluate on the validation data (don't add the second ArgMaxLayer at the end)
              val dfv = model.predict(validationDF, preprocessor, bigdl, false)
              val validationScores = evaluate(dfv, labels.size, conf, "valid")
              content = Serialization.writePretty(validationScores) + ",\n"
              Files.write(Paths.get(scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
            }
          }
        case "ged-lstm" =>
          val what = config.modelType // {tk, ch}
          val langs = Seq("czech", "english", "german", "italian", "swedish")
          for (lang <- langs) {
            val (e, r, j) = if (what == "tk") (config.embeddingSize, config.recurrentSize, config.layers) else (-1, config.recurrentSize, config.layers) // ch
            val conf = Config(modelType = what, embeddingSize = e, recurrentSize = r, layers = j, language = lang, ged = true, batchSize = config.batchSize, 
              driverMemory = config.driverMemory, executorMemory = config.executorMemory)
            val (trainPath, validPath) = dataPaths(conf.language)
            val Array(trainingDF, validationDF) = Array(DataReader.readDataGED(sc, trainPath), DataReader.readDataGED(sc, validPath))
            // create a model
            val model = ModelFactory(conf)
            val prefix = s"${conf.modelPath}/${lang}/${config.modelType}"
            val trainingSummary = TrainSummary(appName = conf.modelType, logDir = s"sum/${lang}/")
            val validationSummary = ValidationSummary(appName = conf.modelType, logDir = s"sum/${lang}/")

            val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
            val bigdl = train(model, conf, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
            // save the model
            preprocessor.write.overwrite.save(s"${prefix}/pre/")
            logger.info("Saving the model...")
            bigdl.saveModel(prefix + "/vsc.bigdl", overWrite = true)

            // evaluate on the training data
            val dft = model.predict(trainingDF, preprocessor, bigdl, true)
            val trainingScores = evaluate(dft, labels.size, conf, "train")
            var content = Serialization.writePretty(trainingScores) + ",\n"
            Files.write(Paths.get(conf.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
            // evaluate on the validation data (don't add the second ArgMaxLayer at the end)
            val dfv = model.predict(validationDF, preprocessor, bigdl, false)
            val validationScores = evaluate(dfv, labels.size, conf, "valid")
            content = Serialization.writePretty(validationScores) + ",\n"
            Files.write(Paths.get(conf.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
          }
        case "ged-bert" =>
          val langs = Seq("czech", "english", "german", "italian", "swedish")
          // use the same BERT config for all languages
          val bertConfig = ConfigBERT(64, 4, 4, config.maxSequenceLength, config.bert.intermediateSize)
          for (lang <- langs) {            
            val conf = Config(modelType = "tb", language = lang, ged = true, bert = bertConfig, batchSize = config.batchSize,
              driverMemory = config.driverMemory, executorMemory = config.executorMemory)
            val (trainPath, validPath) = dataPaths(conf.language)
            val Array(trainingDF, validationDF) = Array(DataReader.readDataGED(sc, trainPath), DataReader.readDataGED(sc, validPath))
            // create a model
            val model = ModelFactory(conf)
            val prefix = s"${conf.modelPath}/${lang}/${conf.modelType}"
            val trainingSummary = TrainSummary(appName = conf.modelType, logDir = s"sum/${lang}/")
            val validationSummary = ValidationSummary(appName = conf.modelType, logDir = s"sum/${lang}/")

            val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
            val bigdl = train(model, conf, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
            // save the model
            preprocessor.write.overwrite.save(s"${prefix}/pre/")
            logger.info("Saving the model...")
            bigdl.saveModel(prefix + "/vsc.bigdl", overWrite = true)

            // evaluate on the training data
            val dft = model.predict(trainingDF, preprocessor, bigdl, true)
            val trainingScores = evaluate(dft, labels.size, conf, "train")
            var content = Serialization.writePretty(trainingScores) + ",\n"
            Files.write(Paths.get(conf.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
            // evaluate on the validation data (don't add the second ArgMaxLayer at the end)
            val dfv = model.predict(validationDF, preprocessor, bigdl, false)
            val validationScores = evaluate(dfv, labels.size, conf, "valid")
            content = Serialization.writePretty(validationScores) + ",\n"
            Files.write(Paths.get(conf.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
          }
        case "predict" =>
          val prefix = s"${config.modelPath}/${inp}/${config.modelType}"
          logger.info(s"Loading preprocessor ${prefix}/pre/...")
          val preprocessor = PipelineModel.load(s"${prefix}/pre/")
          logger.info(s"Loading model ${prefix}/vsc.bigdl...")
          var bigdl = Models.loadModel[Float](prefix + "/vsc.bigdl")
          val labels = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
          val model = ModelFactory(config)
          // read test data and predict
          val testDF = DataReader.readTestDataGED(sc, testPath(config.language))
          val testResult = model.predict(testDF, preprocessor, bigdl, true)
          testResult.show()
          // use "xs" column to find the number of tokens in an input sentence -- this number 
          // can larger than config.maxSeqLength. In this case, we set all predicted labels as correct (1.0)
          val ys = testResult.rdd.map{ case row => (row.getAs[Seq[Float]](0).toArray, row.getAs[Seq[String]](2).size)}
            .map { case (prediction, n) =>
              val zs = prediction.map(k => labels(k.toInt - 1))
              if (n <= zs.size) zs.take(n) else {
                // pad with correct labels
                zs ++ List.fill(n - zs.size)(labels(0))
              }
            }.collect()
          val tokenizer = new RegexTokenizer().setInputCol("x").setOutputCol("t").setPattern("""[\s]+""").setToLowercase(false)
          val us = tokenizer.transform(testResult).select("t").rdd.map{ case row => row.getAs[Seq[String]](0).toArray }.collect()
          val vs = us.zip(ys).map{ case (u, y) => 
            u.zip(y).map{ case (a, b) => a + "\t" + b}.mkString("\n") + "\n"
          }
          val content = vs.mkString("\n")
          Files.write(Paths.get(config.outputPath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)

        case "submission" => // train the character-based model on all the languages with the best hyper-parameters selected by the validation set          
          val langs = Seq("czech", "english", "german", "italian", "swedish")
          val hyperparams = Map("czech" -> (64, 2), "english" -> (128, 2), "german" -> (256, 1), "italian" -> (256, 2), "swedish" -> (256, 2))
          for (lang <- langs) {
            val (r, j) = hyperparams(lang)
            val conf = Config(modelType = "ch", recurrentSize = r, layers = j, language = lang, ged = true, batchSize = config.batchSize, 
              driverMemory = config.driverMemory, executorMemory = config.executorMemory)
            val (trainPath, validPath) = dataPaths(conf.language)
            val Array(trainingDF, validationDF) = Array(DataReader.readDataGED(sc, trainPath), DataReader.readDataGED(sc, validPath))
            // combine training and validation dataset into one df to train
            val df = trainingDF.union(validationDF)
            // create a model
            val model = ModelFactory(conf)
            val prefix = s"${conf.modelPath}/${lang}/${conf.modelType}"
            val trainingSummary = TrainSummary(appName = conf.modelType, logDir = s"sum/${lang}/")
            val validationSummary = ValidationSummary(appName = conf.modelType, logDir = s"sum/${lang}/")

            val (preprocessor, vocabulary, labels) = model.preprocessor(df)
            val bigdl = train(model, conf, df, df, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
            // save the model
            preprocessor.write.overwrite.save(s"${prefix}/pre/")
            logger.info("Saving the model...")
            bigdl.saveModel(prefix + "/vsc.bigdl", overWrite = true)

            // evaluate on the training data
            val dft = model.predict(df, preprocessor, bigdl, true)
            val trainingScores = evaluate(dft, labels.size, conf, "train")
            var content = Serialization.writePretty(trainingScores) + ",\n"
            Files.write(Paths.get(conf.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
          }          
      }
      sc.stop()
      case None => {}
    }
  }
}
