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
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.Matrix

case class ConfigNER(
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y ==> there are Y/X executors 
  executorMemory: String = "8g", // Z
  driverMemory: String = "16g", // D
  mode: String = "eval",
  batchSize: Int = 128,
  epochs: Int = 30,
  learningRate: Double = 5E-4, 
  modelPath: String = "bin/med/",
  trainPath: String = "dat/med/syll.txt",
  validPath: String = "dat/med/val/", // Parquet file of devPath
  outputPath: String = "out/med/",
  scorePath: String = "dat/med/scores-med.json",
  modelType: String = "d", 
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
  * 
  */

object NER {
  implicit val formats = Serialization.formats(NoTypeHints)
  val labelIndex = Map[String, Int](
    "O" -> 0, "B-problem" -> 1, "I-problem" -> 2, "B-treatment" -> 3, "I-treatment" -> 4, "B-test" -> 5, "I-test" -> 6
  )

  def train(config: ConfigNER, trainingDF: DataFrame, developmentDF: DataFrame): PipelineModel = {
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
    val embeddings = config.modelType match {
      case "b" => BertEmbeddings.pretrained("bert_base_multilingual_cased", "xx").setInputCols("document", "token").setOutputCol("embeddings")
      case "x" => XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_afriberta_base", "xx").setInputCols("document", "token").setOutputCol("embeddings")
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
      .setLr(config.learningRate.toFloat).setPo(0.005f)
      .setBatchSize(config.batchSize).setRandomSeed(0)
      .setVerbose(0)
      .setValidationSplit(0.2f)
      // .setEvaluationLogExtended(false).setEnableOutputLogs(false).setIncludeConfidence(true)
      .setEnableMemoryOptimizer(true)
      .setTestDataset(config.validPath)
    val pipeline = new Pipeline().setStages(stages ++ Array(tagger))
    val model = pipeline.fit(trainingDF)
    return model
  }

  def evaluate(result: DataFrame, config: ConfigNER, split: String): ScoreNER = {
    val predictionsAndLabels = result.rdd.map { case row => 
      (row.getAs[Seq[Double]](0).toArray, row.getAs[Seq[Double]](1).toArray)
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

  def saveScore(score: ScoreNER, path: String) = {
    var content = Serialization.writePretty(score) + ",\n"
    Files.write(Paths.get(path), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
  }

  /**
    * Exports result data frame (2-col format) into a text file of CoNLL-2003 format for 
    * evaluation with CoNLL evaluation script (correct <space> prediction).
    * @param result a data frame of two columns "prediction, target"
    * @param config
    * @param split
    */
  def export(result: DataFrame, config: ConfigNER, split: String) = {
    val spark = SparkSession.getActiveSession.get
    import spark.implicits._
    val ss = result.map { row => 
      val prediction = row.getSeq[String](0)
      val target = row.getSeq[String](1)
      val lines = target.zip(prediction).map(p => p._1 + " " + p._2)
      lines.mkString("\n") + "\n"
    }.collect()
    val s = ss.mkString("\n")
    Files.write(Paths.get(s"${config.outputPath}/${config.modelType}-${split}.txt"), s.getBytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
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
        import spark.implicits._
        sc.setLogLevel("ERROR")

        val df = CoNLL(conllLabelIndex = 3).readDatasetFromLines(Source.fromFile(config.trainPath, "UTF-8").getLines.toArray, spark).toDF
        println(s"Number of samples = ${df.count}")
        val Array(trainingDF, developmentDF) = df.randomSplit(Array(0.9, 0.1), 220712L)
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
            validResult.show(5, false)
            // export to CoNLL format
            export(af.select("zs", "ys"), config, "train")
            export(bf.select("zs", "ys"), config, "valid")
        }

        sc.stop()
      case None => {}
    }

  }
}