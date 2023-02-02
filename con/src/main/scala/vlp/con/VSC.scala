package vlp.con

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.keras.models.{Models, KerasNet}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.TimeDistributedCriterion
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.optim.{Loss, Trigger}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.nnframes.{NNModel, NNEstimator}
import com.intel.analytics.bigdl.dllib.nn.{TimeDistributedCriterion, ClassNLLCriterion}

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.linalg.Vector

import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession, DataFrame}

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

import org.apache.log4j.{Logger, Level}
import org.slf4j.LoggerFactory
import java.nio.file.{Files, Paths, StandardOpenOption}


object VSC {
  
  def dataPaths(language: String): (String, String) = {
    language match {
      case "czech" => ("dat/ged/czech/cs_gecc_train.tsv", "dat/ged/czech/cs_gecc_dev.tsv")
      case "english" => ("dat/ged/english/en_fce_train.tsv", "dat/ged/english/en_fce_dev.tsv")
      case "german" => ("dat/ged/german/de_falko-merlin_train.tsv", "dat/ged/german/de_falko-merlin_dev.tsv")
      case "italian" => ("dat/ged/italian/it_merlin_train.tsv", "dat/ged/italian/it_merlin_dev.tsv")
      case "swedish" => ("dat/ged/swedish/sv_swell_train.tsv", "dat/ged/swedish/sv_swell_dev.tsv")
      case _ => ("dat/vsc/100.txt.inp", "dat/vsc/100.txt.inp")
    }
  }

  def evaluate(result: DataFrame, labelSize: Int, config: Config, split: String): Score = {
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
    val precisionByLabel = Array.fill(labelSize)(0d)
    val recallByLabel = Array.fill(labelSize)(0d)
    val fMeasureByLabel = Array.fill(labelSize)(0d)
    // precision by label: our BigDL model uses 1-based labels, so we need to decrease 1 unit.
    val ls = metrics.labels
    println(ls.mkString(", "))
    ls.foreach { k => 
      precisionByLabel(k.toInt-1) = metrics.precision(k)
      recallByLabel(k.toInt-1) = metrics.recall(k)
      fMeasureByLabel(k.toInt-1) = metrics.fMeasure(k)
    }
    val inp = if (config.ged) config.language else config.inputPath.split("/").last.split("""\.""").head
    Score(
      inp, config.modelType, split,
      if (Seq("tk", "st").contains(config.modelType)) config.embeddingSize else -1,
      if (Seq("tb", "sb").contains(config.modelType)) config.bert.hiddenSize else config.recurrentSize,
      if (Seq("tb", "sb").contains(config.modelType)) config.bert.nHead else config.layers,
      metrics.confusionMatrix, metrics.accuracy, precisionByLabel, recallByLabel, fMeasureByLabel
    )
  }

  def train(model: AbstractModel, config: Config, trainingDF: DataFrame, validationDF: DataFrame, 
    preprocessor: PipelineModel, vocabulary: Array[String], labels: Array[String], 
    trainingSummary: TrainSummary, validationSummary: ValidationSummary): KerasNet[Float] = {
    val bigdl = model.createModel(vocabulary.size, labels.size)
    bigdl.summary()
    
    val vocabDict = vocabulary.zipWithIndex.toMap
    val (aft, afv) = (preprocessor.transform(trainingDF), preprocessor.transform(validationDF))
    // map the label to one-based index (for use in ClassNLLCriterion of BigDL), label padding value is -1.
    val labelDict = labels.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    val ySequencer = new Sequencer(labelDict, config.maxSequenceLength, -1).setInputCol("ys").setOutputCol("label")
    val (bft, bfv) = (ySequencer.transform(aft), ySequencer.transform(afv))

    val xSequencer = config.modelType match {
      case "tk" => new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
      case "tb" => new Sequencer4BERT(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
      case "st" => new SubtokenSequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("ts").setOutputCol("features")
      case _ => new CharSequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
    }
    val (cft, cfv) = (xSequencer.transform(bft), xSequencer.transform(bfv))
    cfv.printSchema()

    // our classes are unbalanced, hence we use weights to improve accuracy
    // the first label is more common than al the rest, hence it takes a lesser weight
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
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode, default is false")
      opt[Unit]('g', "GED").action((_, conf) => conf.copy(ged = true)).text("GED mode, default is false")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        implicit val formats = Serialization.formats(NoTypeHints)
        logger.info(Serialization.writePretty(config))

        val conf = Engine.createSparkConf().setAppName(getClass().getName()).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        Engine.init

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
        val prefix = s"${config.modelPath}/${inp}/${config.modelType}"
        val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/${inp}/")
        val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/${inp}/")

        config.mode match {
          case "train" => 
            val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)  
            val bigdl = train(model, config, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
            // save the model
            preprocessor.write.overwrite.save(s"${prefix}/pre/")
            logger.info("Saving the model...")        
            bigdl.saveModel(prefix + "/vsc.bigdl", overWrite = true)

            val trainingAccuracy = trainingSummary.readScalar("TimeDistributedTop1Accuracy")
            val validationLoss = validationSummary.readScalar("Loss")
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
        case "experiment" => 
          // perform multiple experiments for a given language
          // Two models (LSTM, BERT) and two representations (token, subtoken) are run and compared.
          // Different hyper-parameters are tried.
          // 1. LSTM experiments
          val embeddingSizes = Seq(16, 32, 64)
          val recurrentSizes = Seq(32, 64, 128)
          val layerSizes = Seq(1, 2, 3)
          val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
          for (e <- embeddingSizes; r <- recurrentSizes; j <- layerSizes) {
            // each config will be run 3 times
            for (k <- 0 to 2) {
              val conf = Config(embeddingSize = e, recurrentSize = r, layers = j, language = config.language)
              logger.info(Serialization.writePretty(conf))
              val model = ModelFactory(conf)
              val bigdl = train(model, conf, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
              // evaluate on the training data
              val dft = model.predict(trainingDF, preprocessor, bigdl, true)
              val trainingScores = evaluate(dft, labels.size, conf, "train")
              logger.info(s"Training score: ${Serialization.writePretty(trainingScores)}") 
              var content = Serialization.writePretty(trainingScores) + ",\n"
              Files.write(Paths.get(conf.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
              // evaluate on the validation data (don't add the second ArgMaxLayer at the end)
              val dfv = model.predict(validationDF, preprocessor, bigdl, false)
              val validationScores = evaluate(dfv, labels.size, conf, "valid")
              logger.info(s"Validation score: ${Serialization.writePretty(validationScores)}")
              content = Serialization.writePretty(validationScores) + ",\n"
              Files.write(Paths.get(conf.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
            }       
          }
      }
      sc.stop()
      case None => {}
    }
  }
}
