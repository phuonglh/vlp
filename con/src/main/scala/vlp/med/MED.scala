package vlp.med

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach
import com.johnsnowlabs.nlp.embeddings.XlmRoBertaSentenceEmbeddings
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, length, not, trim}
import org.json4s.{Formats, NoTypeHints}
import org.json4s.jackson.Serialization
import scopt.OptionParser

import java.nio.file.{Files, Paths}

object MED {
  def readCorpus(spark: SparkSession, language: String): DataFrame = {
    val trainPath = s"dat/med/ntcir17_mednlp-sc_sm_train_26_06_23/${language}.csv"
    import spark.implicits._
    val df = spark.read.text(trainPath).filter(length(trim(col("value"))) > 0)
    val header = df.first().get(0)
    df.filter(not(col("value").contains(header))).map { row =>
      val line = row.getAs[String](0)
      val j = line.indexOf(",")
      if (j <= 1 || j > 8) println("ERROR: " + line)
      val n = line.length
      val trainId = line.substring(0, j).toDouble
      val text = line.substring(j+1, n-43)
      val ys = line.substring(n-43) // 22 binary labels, plus 21 commas
      val labels = ys.split(",").zipWithIndex.filter(_._1 == "1").map(_._2.toString)
      (trainId, text, if (labels.length > 0) labels else Array("NA"))
    }.toDF("id", "text", "ys")
  }

  def createPipeline(df: DataFrame, config: Config): PipelineModel = {
    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenEmbeddings = XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_base", "xx").setInputCols("document").setOutputCol("embeddings")
    //    val labelIndexer = new StringIndexer().setInputCol("ys").setOutputCol("label")
    val classifier = new MultiClassifierDLApproach().setInputCols("embeddings").setOutputCol("category").setLabelColumn("ys")
      .setBatchSize(config.batchSize).setMaxEpochs(config.epochs).setLr(config.learningRate.toFloat)
    val validPath = s"dat/med/${config.language}"
    if (!Files.exists(Paths.get(validPath))) {
      val preprocessor = new Pipeline().setStages(Array(documentAssembler, tokenEmbeddings))
      val preprocessorModel = preprocessor.fit(df)
      val ef = preprocessorModel.transform(df)
      ef.write.save(validPath)
    }
    classifier.setTestDataset(validPath)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenEmbeddings, classifier))
    pipeline.fit(df)
  }

  def evaluate(result: DataFrame, config: Config, split: String): Score = {
    val predictionsAndLabels = result.rdd.map { row =>
      (row.getAs[Seq[Double]](0).toArray, row.getAs[Seq[Double]](1).toArray)
    }
    val metrics = new MultilabelMetrics(predictionsAndLabels)
    val ls = metrics.labels
    val numLabels = ls.max.toInt + 1 // zero-based labels
    val precisionByLabel = Array.fill(numLabels)(0d)
    val recallByLabel = Array.fill(numLabels)(0d)
    val fMeasureByLabel = Array.fill(numLabels)(0d)
    println(ls.mkString(", "))
    ls.foreach { k =>
      precisionByLabel(k.toInt) = metrics.precision(k)
      recallByLabel(k.toInt) = metrics.recall(k)
      fMeasureByLabel(k.toInt) = metrics.f1Measure(k)
    }
    Score(config.language, split, metrics.accuracy, metrics.f1Measure, precisionByLabel, recallByLabel, fMeasureByLabel)
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[Config](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 8g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language, either en/fr/de/ja")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val spark = SparkSession.builder().config("spark.driver.memory", config.driverMemory).master(config.master).appName("MED").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        implicit val formats: AnyRef with Formats = Serialization.formats(NoTypeHints)
        println(Serialization.writePretty(config))
        val df = readCorpus(spark, config.language)
        df.show()
        df.printSchema()
        println(s"Number of samples = ${df.count}")
        val model = createPipeline(df, config)
        val ef = model.transform(df)
        ef.show()
        ef.printSchema()
        val score = evaluate(ef, config, "train")
        println(Serialization.writePretty(score))

        spark.stop()
      case None =>
    }
  }
}
