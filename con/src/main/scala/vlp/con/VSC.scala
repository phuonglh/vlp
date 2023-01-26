package vlp.con

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.keras.models.Models
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.TimeDistributedCriterion
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.optim.Loss
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.nnframes.{NNModel, NNEstimator}
import com.intel.analytics.bigdl.dllib.optim.Trigger
import com.intel.analytics.bigdl.dllib.nn.{TimeDistributedCriterion, ClassNLLCriterion}

import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.linalg.Vector


import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

import org.apache.log4j.{Logger, Level}
import org.slf4j.LoggerFactory


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
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()    
    import spark.implicits._
    val df = sc.textFile(config.inputPath).zipWithIndex.toDF("line", "id")
    val df0 = df.filter(col("id") % 2 === 0).withColumn("y", col("line"))
    val df1 = df.filter(col("id") % 2 === 1).withColumn("x", col("line")).withColumn("id0", col("id") - 1)
    val af = df0.join(df1, df0.col("id") === df1.col("id0"))
    return af.select("x", "y")
  }
  
  def evaluate(result: DataFrame, labelSize: Int): Score = {
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
    // precision by label
    val ls = metrics.labels
    ls.foreach { k => 
      precisionByLabel(k.toInt-1) = metrics.precision(k)
      recallByLabel(k.toInt-1) = metrics.recall(k)
      fMeasureByLabel(k.toInt-1) = metrics.fMeasure(k)
    }
    Score(metrics.confusionMatrix, metrics.accuracy, precisionByLabel, recallByLabel, fMeasureByLabel)
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
      opt[Double]('n', "percentage").action((x, conf) => conf.copy(percentage = x)).text("percentage of the data set to use")
      opt[Double]('u', "dropoutProbability").action((x, conf) => conf.copy(dropoutProbability = x)).text("dropout ratio")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('l', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("max sequence length")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 0.001")
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
        // split data
        val Array(trainingDF, validationDF) = df.randomSplit(Array(0.8, 0.2), seed = 85L)
        // create a model
        val model = ModelFactory(config.modelType, config)
        config.mode match {
          case "train" => 
            val (preprocessor, vocabulary, labels) = model.preprocessor(trainingDF)  
            val bigdl = model.createModel(vocabulary.size, labels.size)
            bigdl.summary()
            
            val vocabDict = vocabulary.zipWithIndex.toMap
            val (aft, afv) = (preprocessor.transform(trainingDF), preprocessor.transform(validationDF))
            // map the label to one-based index (for use in ClassNLLCriterion of BigDL), label padding value is -1.
            val labelDict = labels.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
            val ySequencer = new Sequencer(labelDict, config.maxSequenceLength, -1).setInputCol("ys").setOutputCol("label")
            val (bft, bfv) = (ySequencer.transform(aft), ySequencer.transform(afv))

            val (cft, cfv) = config.modelType match {
              case "tk" => 
                val xSequencer = new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
                (xSequencer.transform(bft), xSequencer.transform(bfv))
              case "tb" =>
                val xSequencer = new Sequencer4BERT(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
                (xSequencer.transform(bft), xSequencer.transform(bfv))
              case _ => 
                val xSequencer = new MultiHotSequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
                (xSequencer.transform(bft), xSequencer.transform(bfv))
            }
            cfv.printSchema()

            // get the input data set name (for example, "vud", "fin") 
            val inp = config.inputPath.split("/").last.split("""\.""").head
            val prefix = s"${config.modelPath}/${inp}/${config.modelType}"

            val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/${inp}/")
            val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/${inp}/")

            // our classes are unbalanced, hence we use weights to improve accuracy
            val w = Tensor(Array(labelDict.size)).rand()
            w.setValue(1, 0.1f); for (j <- 2 to 5) w.setValue(j, 0.9f)

            val maxSeqLen = config.maxSequenceLength
            val classifier = config.modelType match {
              case "tk" => 
                val (featureSize, labelSize) = (Array(maxSeqLen), Array(maxSeqLen))
                NNEstimator(bigdl, TimeDistributedCriterion(ClassNLLCriterion(weights=w, logProbAsInput=false), true), featureSize, labelSize)
              case "tb" => 
                val (featureSize, labelSize) = (Array(Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen)), Array(maxSeqLen))
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
            // at the end to output a good format for BigDL. See the predict() method.
            classifier.fit(cft)

            val trainingAccuracy = trainingSummary.readScalar("TimeDistributedTop1Accuracy")
            val validationLoss = validationSummary.readScalar("Loss")
            val validationAccuracy = validationSummary.readScalar("TimeDistributedTop1Accuracy")
            logger.info("Train Accuracy: " + trainingAccuracy.mkString(", "))
            logger.info("Valid Accuracy: " + validationAccuracy.mkString(", "))
            // evaluate             
            // val dft = model.predict(trainingDF, preprocessor, bigdl)
            // val trainingScores = evaluate(dft, labels.size)
            // logger.info(s"Training score: ${Serialization.writePretty(trainingScores)}") 
            val dfv = model.predict(validationDF, preprocessor, bigdl)
            val validationScores = evaluate(dfv, labels.size)
            logger.info(s"Validation score: ${Serialization.writePretty(validationScores)}")
            // // save the model
            // preprocessor.write.overwrite.save(s"${prefix}/pre/")
            // logger.info("Saving the model...")        
            // bigdl.saveModel(prefix + "/vsc.bigdl", overWrite = true)
        case "eval" => 
          val inp = config.inputPath.split("/").last.split("""\.""").head
          val prefix = s"${config.modelPath}/${inp}/${config.modelType}"
          logger.info(s"Loading preprocessor ${prefix}/pre/...")
          val preprocessor = PipelineModel.load(s"${prefix}/pre/")
          logger.info(s"Loading model ${prefix}/vsc.bigdl...")
          val bigdl = Models.loadModel[Float](prefix + "/vsc.bigdl")
          val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
          val labelDict = labels.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
          val Array(trainingDF, validationDF) = df.randomSplit(Array(0.8, 0.2), seed = 85L)
          val model = ModelFactory(bigdl, config)
          val result = model.predict(validationDF, preprocessor, bigdl)
          val scores = evaluate(result, labels.size)
          logger.info(Serialization.writePretty(scores))
        }
        sc.stop()
      case None => {}
    }
  }
}
