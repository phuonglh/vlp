package vlp.con

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.embeddings.{BertEmbeddings, DeBertaEmbeddings, DistilBertEmbeddings}
import com.johnsnowlabs.nlp.{DocumentAssembler, EmbeddingsFinisher}
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.training.CoNLL
import scala.io.Source

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

import java.nio.file.{Files, Paths, StandardOpenOption}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.ml.linalg.DenseVector


import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.models.{Models, KerasNet}
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.nnframes.{NNModel, NNEstimator}
import com.intel.analytics.bigdl.dllib.nn.{TimeDistributedCriterion, ClassNLLCriterion}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.optim.Trigger


case class ConfigNER(
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y
  executorMemory: String = "8g", // Z
  driverMemory: String = "16g", // D
  mode: String = "eval",
  batchSize: Int = 128,
  maxSeqLen: Int = 80,
  hiddenSize: Int = 64,
  epochs: Int = 30,
  learningRate: Double = 5E-4, 
  modelPath: String = "bin/med/",
  trainPath: String = "dat/med/syll.txt",
  validPath: String = "dat/med/val/", // Parquet file of devPath
  outputPath: String = "out/med/",
  scorePath: String = "dat/med/scores-med.json",
  modelType: String = "s", 
)

case class ScoreNER(
  modelType: String,
  split: String,
  accuracy: Double,
  confusionMatrix: Matrix,
  precision: Array[Double],
  recall: Array[Double],
  fMeasure: Array[Double]
)

/**
  * phuonglh, April 2023
  * 
  * An implementation of Vietnamese NER on a medical data set using pretrained models. 
  * 
  */

object NER {
  implicit val formats: AnyRef with Formats = Serialization.formats(NoTypeHints)
  private val labelIndex = Map[String, Int](
    "O" -> 1, "B-problem" -> 2, "I-problem" -> 3, "B-treatment" -> 4, "I-treatment" -> 5, "B-test" -> 6, "I-test" -> 7
  )
  val labelDict: Map[Double, String] = labelIndex.keys.map(k => (labelIndex(k).toDouble, k)).toMap

  /**
    * Trains a NER model using the BigDL framework with user-defined model. This approach is more flexible than the [[trainJSL()]] method.
    *
    * @param config a config
    * @param trainingDF a training df
    * @param developmentDF a development df
    * @return a preprocessor and a BigDL model
    */
  private def trainBDL(config: ConfigNER, trainingDF: DataFrame, developmentDF: DataFrame): (PipelineModel, KerasNet[Float]) = {
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
    val embeddings = config.modelType match {
      case "b" => BertEmbeddings.pretrained("bert_base_multilingual_cased", "xx").setInputCols("document", "token").setOutputCol("embeddings")
      case "d" => DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true)
      case "m" => DeBertaEmbeddings.pretrained("mdeberta_v3_base", "xx").setInputCols("document", "token").setOutputCol("embeddings")
      case "s" => DistilBertEmbeddings.pretrained("distilbert_base_cased", "vi").setInputCols("document", "token").setOutputCol("embeddings")
      case "x" => XlmRoBertaEmbeddings.pretrained("xlm_roberta_large", "xx").setInputCols("document", "token").setOutputCol("embeddings") // _large / _base
      case _ => DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true)
    }
    val finisher = new EmbeddingsFinisher().setInputCols("embeddings").setOutputCols("xs").setOutputAsVector(false) // output as arrays
    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings, finisher))
    val preprocessor = pipeline.fit(trainingDF)
    val (af, bf) = (preprocessor.transform(trainingDF), preprocessor.transform(developmentDF))
    // supplement pipeline for BigDL
    val bigdlPreprocessor = pipelineBigDL(config).fit(af)
    val (uf, vf) = (bigdlPreprocessor.transform(af), bigdlPreprocessor.transform(bf))
    // create a BigDL model
    val bigdl = Sequential()
    bigdl.add(InputLayer(inputShape = Shape(config.maxSeqLen*768)).setName("input"))
    bigdl.add(Reshape(targetShape=Array(config.maxSeqLen, 768)).setName("reshape"))
    bigdl.add(Bidirectional(LSTM(outputDim = config.hiddenSize, returnSequences = true).setName("LSTM")))
    bigdl.add(Dropout(0.1).setName("dropout"))
    bigdl.add(Dense(labelIndex.size, activation="softmax").setName("dense"))
    val (featureSize, labelSize) = (Array(config.maxSeqLen*768), Array(config.maxSeqLen))
    // should set the sizeAverage=false in ClassNLLCriterion
    val estimator = NNEstimator(bigdl, TimeDistributedCriterion(ClassNLLCriterion(logProbAsInput = false, sizeAverage = false), sizeAverage = true), featureSize, labelSize)
    val trainingSummary = TrainSummary(appName = config.modelType, logDir = "sum/med/")
    val validationSummary = ValidationSummary(appName = config.modelType, logDir = "sum/med/")
    estimator.setLabelCol("target").setFeaturesCol("features")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(config.learningRate))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, vf, Array(new TimeDistributedTop1Accuracy(paddingValue = -1)), config.batchSize)
    estimator.fit(uf)
    (preprocessor, bigdl)
  }

  /**
    * Builds a pipeline for BigDL model: sequencer -> flattener -> padder
    *
    * @param config config
    */
  private def pipelineBigDL(config: ConfigNER): Pipeline = {
    // use a label sequencer to transform `ys` into sequences of integers (one-based, for BigDL to work)
    val sequencer = new Sequencer(labelIndex, config.maxSeqLen, -1f).setInputCol("ys").setOutputCol("target")
    val flattener = new FeatureFlattener().setInputCol("xs").setOutputCol("as")
    val padder = new FeaturePadder(config.maxSeqLen*768, 0f).setInputCol("as").setOutputCol("features")
    new Pipeline().setStages(Array(sequencer, flattener, padder))
  }

  /**
    * Trains a NER model using the JohnSnowLab [[NerDLApproach]]. This is a CNN-BiLSTM-CRF network model, which is readily usable but not 
    * flexible enough.
    *
    * @param config a config
    * @param trainingDF a df
    * @param developmentDF a df
    */
  private def trainJSL(config: ConfigNER, trainingDF: DataFrame, developmentDF: DataFrame): PipelineModel = {
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
    val embeddings = config.modelType match {
      case "m" => DeBertaEmbeddings.pretrained("mdeberta_v3_base", "xx").setInputCols("document", "token").setOutputCol("embeddings")
      case "d" => DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true)
      case "s" => DistilBertEmbeddings.pretrained("distilbert_base_cased", "vi").setInputCols("document", "token").setOutputCol("embeddings")
      case _ => DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("embeddings").setCaseSensitive(true)
    }
    // val finisher = new EmbeddingsFinisher().setInputCols("embeddings").setOutputCols("xs").setOutputAsVector(true).setCleanAnnotations(false)
    val stages = Array(document, tokenizer, embeddings)
    // train a preprocessor 
    val preprocessor = new Pipeline().setStages(stages)
    val preprocessorModel = preprocessor.fit(trainingDF)
    // use the preprocessor pipeline to transform the data sets
    val df = preprocessorModel.transform(developmentDF)
    df.write.mode("overwrite").parquet(config.validPath)
    val tagger = new NerDLApproach().setInputCols(Array("document", "token", "embeddings"))
      .setLabelColumn("label").setOutputCol("ner")
      .setMaxEpochs(config.epochs)
      .setLr(config.learningRate.toFloat).setPo(0.005f)
      .setBatchSize(config.batchSize).setRandomSeed(0)
      .setVerbose(0)
      .setValidationSplit(0.2f)
      // .setEvaluationLogExtended(false).setEnableOutputLogs(false).setIncludeConfidence(true)
      .setEnableMemoryOptimizer(true)
      .setTestDataset(config.validPath)
    val pipeline = new Pipeline().setStages(stages ++ Array(tagger))
    val model = pipeline.fit(trainingDF)
    model
  }

  def evaluate(result: DataFrame, config: ConfigNER, split: String): ScoreNER = {
    val predictionsAndLabels = result.rdd.map { row =>
      val zs = row.getAs[Seq[Float]](0).map(_.toDouble).toArray
      val ys = row.getAs[DenseVector](1).toArray
      var j = ys.indexOf(-1f) // first padding value in the label sequence
      if (j < 0) j = ys.length
      val i = Math.min(config.maxSeqLen, j)
      (zs.take(i), ys.take(i))
    }.flatMap { case (prediction, label) => prediction.zip(label) }
    val metrics = new MulticlassMetrics(predictionsAndLabels)
    val ls = metrics.labels
    val numLabels = ls.max.toInt + 1 // zero-based labels
    val precisionByLabel = Array.fill(numLabels)(0d)
    val recallByLabel = Array.fill(numLabels)(0d)
    val fMeasureByLabel = Array.fill(numLabels)(0d)
    ls.foreach { k => 
      precisionByLabel(k.toInt) = metrics.precision(k)
      recallByLabel(k.toInt) = metrics.recall(k)
      fMeasureByLabel(k.toInt) = metrics.fMeasure(k)
    }
    ScoreNER(
      config.modelType, split,
      metrics.accuracy, metrics.confusionMatrix, 
      precisionByLabel, recallByLabel, fMeasureByLabel
    )
  }

  private def saveScore(score: ScoreNER, path: String) = {
    val content = Serialization.writePretty(score) + ",\n"
    Files.write(Paths.get(path), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
  }

  /**
    * Exports result data frame (2-col format) into a text file of CoNLL-2003 format for 
    * evaluation with CoNLL evaluation script (correct <space> prediction).
    * @param result a data frame of two columns "prediction, target"
    * @param config a config
    * @param split a split name
    */
  private def export(result: DataFrame, config: ConfigNER, split: String) = {
    val spark = SparkSession.getActiveSession.get
    import spark.implicits._
    val ss = result.map { row => 
      val prediction = row.getSeq[String](0)
      val target = row.getSeq[String](1)
      val lines = target.zip(prediction).map(p => p._1 + " " + p._2)
      lines.mkString("\n") + "\n"
    }.collect()
    val s = ss.mkString("\n")
    Files.write(Paths.get(s"${config.outputPath}/${config.modelType}-$split.txt"), s.getBytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def predict(preprocessor: PipelineModel, bigdl: KerasNet[Float], df: DataFrame, config: ConfigNER, argmax: Boolean=true): DataFrame = {
    val bf = preprocessor.transform(df)
    val bigdlPreprocessor = pipelineBigDL(config).fit(bf)
    val vf = bigdlPreprocessor.transform(bf)
    // convert bigdl to sequential model
    val sequential = bigdl.asInstanceOf[Sequential[Float]]
    // bigdl produces 3-d output results (including batch dimension), we need to convert it to 2-d results.
    if (argmax)
      sequential.add(ArgMaxLayer())
    sequential.summary()
    // wrap to a Spark model and run prediction
    val model = NNModel(sequential)
    model.transform(vf)
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[ConfigNER](getClass.getName) {
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
    opts.parse(args, ConfigNER()) match {
      case Some(config) =>
        val conf = new SparkConf().setAppName(getClass.getName).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        Engine.init
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        sc.setLogLevel("ERROR")
        // read the df using the CoNLL format of Spark-NLP, which provides some columns, including [text, label] columns.
        val df = CoNLL(conllLabelIndex = 3).readDatasetFromLines(Source.fromFile(config.trainPath, "UTF-8").getLines.toArray, spark).toDF
        val af = df.withColumn("ys", col("label.result"))
        println(s"Number of samples = ${df.count}")
        val Array(trainingDF, developmentDF) = af.randomSplit(Array(0.9, 0.1), 220712L)
        developmentDF.show()
        developmentDF.printSchema()
        val modelPath = config.modelPath + "/" + config.modelType
        config.mode match {
          case "train" =>
            // val model = trainJSL(config, trainingDF, developmentDF)
            val (preprocessor, bigdl) = trainBDL(config, trainingDF, developmentDF)
            preprocessor.write.overwrite.save(modelPath)
            bigdl.saveModel(modelPath + "/ner.bigdl", overWrite = true)
            val output = predict(preprocessor, bigdl, developmentDF, config)
            output.show
          case "predict" =>
          case "evalBDL" => 
            val preprocessor = PipelineModel.load(modelPath)
            val bigdl = Models.loadModel[Float](modelPath + "/ner.bigdl")
            // training result
            val outputTrain = predict(preprocessor, bigdl, trainingDF.sample(0.1), config)
            outputTrain.show
            outputTrain.printSchema
            val trainResult = outputTrain.select("prediction", "target")
            var score = evaluate(trainResult, config, "train")
            saveScore(score, config.scorePath)
            // validation result
            val outputValid = predict(preprocessor, bigdl, developmentDF, config, argmax = false)
            outputValid.show
            val validResult = outputValid.select("prediction", "target")
            score = evaluate(validResult, config, "valid")
            saveScore(score, config.scorePath)
            // convert "prediction" column to human-readable label column "zs"
            val sequencerPrediction = new SequencerDouble(labelDict).setInputCol("prediction").setOutputCol("zs")
            val af = sequencerPrediction.transform(outputTrain)
            val bf = sequencerPrediction.transform(outputValid)
            // export to CoNLL format
            export(af.select("zs", "ys"), config, "train")
            export(bf.select("zs", "ys"), config, "valid")
          case "evalJSL" => 
            val model = PipelineModel.load(modelPath)
            val tf = model.transform(trainingDF).withColumn("zs", col("ner.result")).withColumn("ys", col("label.result"))
            val sequencerPrediction = new SequencerNER(labelIndex).setInputCol("zs").setOutputCol("prediction")
            val sequencerTarget = new SequencerNER(labelIndex).setInputCol("ys").setOutputCol("target")
            // training result
            val af = sequencerTarget.transform(sequencerPrediction.transform(tf))
            val trainResult = af.select("prediction", "target")
            var score = evaluate(trainResult, config, "train")
            saveScore(score, config.scorePath)
            // validation result            
            val vf = model.transform(developmentDF).withColumn("zs", col("ner.result")).withColumn("ys", col("label.result"))
            val bf = sequencerTarget.transform(sequencerPrediction.transform(vf))
            val validResult = bf.select("prediction", "target")
            score = evaluate(validResult, config, "valid")
            saveScore(score, config.scorePath)
            validResult.show(5, truncate = false)
            // export to CoNLL format
            export(af.select("zs", "ys"), config, "train")
            export(bf.select("zs", "ys"), config, "valid")
        }

        sc.stop()
      case None =>
    }

  }
}