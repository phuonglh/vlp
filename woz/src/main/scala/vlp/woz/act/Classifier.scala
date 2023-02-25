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

/**
  * phuonglh@gmail.com
  * 
  * February 2023
  * 
  */
object Classifier {
  def train(model: AbstractModel, config: Config, trainingDF: DataFrame, validationDF: DataFrame, 
    preprocessor: PipelineModel, vocabulary: Array[String], labels: Array[String], 
    trainingSummary: TrainSummary, validationSummary: ValidationSummary): KerasNet[Float] = {
    val bigdl = model.createModel(vocabulary.size, labels.size)
    bigdl.summary()
    // build a vocab map
    val vocabDict = vocabulary.zipWithIndex.toMap

    val xSequencer = if (config.modelType == "lstm") {
        new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
    } else { 
      new Sequencer4BERT(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features") 
    }

    val (aft, afv) = (preprocessor.transform(trainingDF), preprocessor.transform(validationDF))
    val (cft, cfv) = (xSequencer.transform(aft), xSequencer.transform(afv))
    cfv.printSchema()

    val maxSeqLen = config.maxSequenceLength
    val classifier = if (config.modelType == "lstm") {
        val (featureSize, labelSize) = (Array(maxSeqLen), Array(labels.size))
        NNEstimator(bigdl, BCECriterion(), featureSize, labelSize)
    } else {
        val featureSize = Array(Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen))
        val labelSize = Array(labels.size)
        NNEstimator(bigdl, BCECriterion(), featureSize, labelSize)
    }

    classifier.setLabelCol("label").setFeaturesCol("features")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(config.learningRate))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, cfv, Array(new CategoricalAccuracy(), new MAE()), config.batchSize)
    // fit the classifier, which will train the bigdl model and return a NNModel
    // but we cannot use this NNModel to transform because we need a custom layer ArgMaxLayer 
    // at the end to output a good format for BigDL. See the predict() method for detail.
    classifier.fit(cft)
    return bigdl
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
    Score(
      config.modelType, split,
      if (config.modelType == "lstm") config.embeddingSize else -1,
      if (config.modelType == "bert") config.bert.hiddenSize else config.recurrentSize,
      if (config.modelType == "bert") config.bert.nBlock else config.layers,
      if (config.modelType == "bert") config.bert.nHead else -1,
      if (config.modelType == "bert") config.bert.intermediateSize else -1,
      metrics.confusionMatrix, metrics.accuracy, precisionByLabel, recallByLabel, fMeasureByLabel
    )
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
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 0.001")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[String]('s', "scorePath").action((x, conf) => conf.copy(scorePath = x)).text("score path")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode, default is false")
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

        // create a model
        val model = ModelFactory(config)
        val prefix = s"${config.modelPath}/${config.modelType}"
        val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/act/${config.modelType}/")
        val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/act/${config.modelType}/")
        // read train/dev datasets
        val (trainingDF, validationDF) = (spark.read.json(config.trainPath), spark.read.json(config.devPath))

        config.mode match {
          case "train" => 
            logger.info(Serialization.writePretty(config))
            val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)
            val bigdl = train(model, config, trainingDF, validationDF, preprocessor, vocabulary, labels, trainingSummary, validationSummary)
            // save the model
            preprocessor.write.overwrite.save(s"${config.modelPath}/pre/")
            logger.info("Saving the model...")        
            bigdl.saveModel(prefix + "/act.bigdl", overWrite = true)
            val trainingAccuracy = trainingSummary.readScalar("Top1Accuracy")
            val validationLoss = validationSummary.readScalar("Loss")
            val validationAccuracy = validationSummary.readScalar("Top1Accuracy")
            logger.info("Train Accuracy: " + trainingAccuracy.mkString(", "))
            logger.info("Valid Accuracy: " + validationAccuracy.mkString(", "))
            logger.info("Validation Loss: " + validationLoss.mkString(", "))
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
            logger.info(s"Loading preprocessor ${config.modelPath}/pre/...")
            val preprocessor = PipelineModel.load(s"${config.modelPath}/pre/")
            logger.info(s"Loading model ${prefix}/act.bigdl...")
            var bigdl = Models.loadModel[Float](prefix + "/act.bigdl")
            val labels = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
            val labelDict = labels.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
            logger.info(labelDict.toString)
            val model = ModelFactory(config)
            val trainingResult = model.predict(trainingDF, preprocessor, bigdl, true)
            trainingResult.show(false)
            var scores = evaluate(trainingResult, labels.size, config, "train")
            logger.info(Serialization.writePretty(scores))
            var content = Serialization.writePretty(scores) + ",\n"
            Files.write(Paths.get(config.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
            val result = model.predict(validationDF, preprocessor, bigdl, false)
            scores = evaluate(result, labels.size, config, "valid")
            logger.info(Serialization.writePretty(scores))
            content = Serialization.writePretty(scores) + ",\n"
            Files.write(Paths.get(config.scorePath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
          case _ => logger.error("What mode do you want to run?")
        }
        sc.stop()
      case None => {}
    }
  }    
}
