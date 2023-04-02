package vlp.con

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.functions._
import com.johnsnowlabs.nlp.embeddings.{BertEmbeddings, DeBertaEmbeddings, DistilBertEmbeddings, XlnetEmbeddings}
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, EmbeddingsFinisher}
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

import java.nio.file.{Files, Paths, StandardOpenOption}
import com.johnsnowlabs.nlp.training.CoNLL
import scala.io.Source
import org.apache.spark.mllib.evaluation.MultilabelMetrics

case class ConfigNER(
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y ==> there are Y/X executors 
  executorMemory: String = "8g", // Z
  driverMemory: String = "16g", // D
  mode: String = "eval",
  batchSize: Int = 128,
  epochs: Int = 10,
  learningRate: Double = 5E-4, 
  modelPath: String = "bin/med/",
  trainPath: String = "dat/med/syll.txt",
  validPath: String = "dat/med/val/", // Parquet file of devPath
  outputPath: String = "dat/out/",
  scorePath: String = "dat/scores-med.json",
  modelType: String = "d", 
)

case class ScoreNER(
  modelType: String,
  split: String,
  accuracy: Double,
  f1Measure: Double,
  microF1Measure: Double, 
  microPrecision: Double,
  microRecall: Double,
  precision: Array[Double],
  recall: Array[Double],
  fMeasure: Array[Double]
)

/**
  * phuonglh, April 2023
  * 
  * 
  */

object NER {
  implicit val formats = Serialization.formats(NoTypeHints)

  def train(config: ConfigNER, trainingDF: DataFrame, developmentDF: DataFrame): PipelineModel = {
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
    val embeddings = config.modelType match {
      case "b" => BertEmbeddings.pretrained().setInputCols("document").setOutputCol("embeddings")
      case "x" => XlmRoBertaEmbeddings.pretrained().setInputCols("document").setOutputCol("embeddings")
      case "d" => DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("embeddings")
      case "s" => DistilBertEmbeddings.pretrained("distilbert_base_cased", "vi").setInputCols("document", "token").setOutputCol("embeddings")
      case _ => DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("embeddings")
    }
    // val finisher = new EmbeddingsFinisher().setInputCols("embeddings").setOutputCols("xs").setOutputAsVector(true).setCleanAnnotations(false)
    var stages = Array(document, tokenizer, embeddings)
    // train a preprocessor 
    val preprocessor = new Pipeline().setStages(stages)
    val preprocessorModel = preprocessor.fit(trainingDF)
    // use the preprocessor pipeline to transform the data sets
    val df = preprocessorModel.transform(developmentDF)
    df.write.mode("overwrite").parquet(config.validPath)
    val tagger = new NerDLApproach().setInputCols(Array("document", "token", "embeddings"))
      .setLabelColumn("label").setOutputCol("ner")
      .setMaxEpochs(config.epochs)
      .setLr(config.learningRate.toFloat)
      .setPo(0.005f)
      .setBatchSize(8)
      .setRandomSeed(0)
      .setVerbose(0)
      .setValidationSplit(0.2f)
      .setEvaluationLogExtended(false).setEnableOutputLogs(false).setIncludeConfidence(true)
      .setTestDataset(config.validPath)
    val pipeline = new Pipeline().setStages(stages ++ Array(tagger))
    val model = pipeline.fit(trainingDF)
    return model
  }

  def evaluate(result: DataFrame, config: ConfigNER, split: String): ScoreNER = {
    val predictionsAndLabels = result.rdd.map { case row => 
      (row.getAs[Seq[Double]](0).toArray, row.getAs[Seq[Double]](1).toArray)
    }
    val metrics = new MultilabelMetrics(predictionsAndLabels)
    val ls = metrics.labels
    val numLabels = ls.max.toInt + 1 // zero-based labels
    val precisionByLabel = Array.fill(numLabels)(0d)
    val recallByLabel = Array.fill(numLabels)(0d)
    val fMeasureByLabel = Array.fill(numLabels)(0d)
    ls.foreach { k => 
      precisionByLabel(k.toInt) = metrics.precision(k)
      recallByLabel(k.toInt) = metrics.recall(k)
      fMeasureByLabel(k.toInt) = metrics.f1Measure(k)
    }
    ScoreNER(
      config.modelType, split,
      metrics.accuracy, metrics.f1Measure, 
      metrics.microF1Measure, metrics.microPrecision, metrics.microRecall,
      precisionByLabel, recallByLabel, fMeasureByLabel
    )
  }

  def saveScore(score: ScoreNER, path: String) = {
    var content = Serialization.writePretty(score) + ",\n"
    Files.write(Paths.get(path), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[ConfigNER](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 16g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
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
        val conf = new SparkConf().setAppName(getClass().getName()).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        sc.setLogLevel("ERROR")

        val df = CoNLL(conllLabelIndex = 3).readDatasetFromLines(Source.fromFile(config.trainPath, "UTF-8").getLines.toArray, spark).toDF
        println(s"Number of samples = ${df.count}")
        val Array(trainingDF, developmentDF) = df.randomSplit(Array(0.8, 0.2), 220712L)
        developmentDF.show()
        developmentDF.printSchema()
        val modelPath = config.modelPath + "/" + config.modelType
        config.mode match {
          case "train" =>
            val model = train(config, trainingDF, developmentDF)
            val output = model.transform(developmentDF)
            output.printSchema
            output.show()
            model.write.overwrite.save(modelPath)
          case "predict" =>
          case "eval" => 
            val tf = trainingDF.withColumn("ys", col("label.result"))
            val vf = developmentDF.withColumn("ys", col("label.result"))
        }

        sc.stop()
      case None => {}
    }

  }
}