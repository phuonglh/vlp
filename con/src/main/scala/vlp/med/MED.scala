package vlp.med

import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler}
import com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach
import com.johnsnowlabs.nlp.embeddings.{BertSentenceEmbeddings, UniversalSentenceEncoder, XlmRoBertaSentenceEmbeddings}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, length, not, trim, udf}
import org.json4s.{Formats, NoTypeHints}
import org.json4s.jackson.Serialization
import scopt.OptionParser

import java.nio.file.{Files, Paths}
import com.johnsnowlabs.nlp.functions._
import org.apache.spark.ml.linalg.Vectors


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
      var text = line.substring(j+1, n-43)
      // remove quotes at the beginning or the end of the text if there are
      if (text.startsWith("\"")) text = text.substring(1, text.size-1)
      val ys = line.substring(n-43) // 22 binary labels, plus 21 commas
      // convert to 23 labels [0, 1, 2,..., 22]. The zero vector corresponds to "0". Other labels are shifted by their index by 1.
      val labels = ys.split(",").zipWithIndex.filter(_._1 == "1").map(p => (p._2 + 1).toString)
      (trainId, text, if (labels.length > 0) labels else Array("0"))
    }.toDF("id", s"text:${language}", s"ys:${language}")
  }

  def readCorpus(spark: SparkSession): DataFrame = {
    val languages = List("de", "fr", "en", "ja")
    val dfs = languages.map(readCorpus(spark, _))
    dfs.reduce((df1, df2) => df1.join(df2, "id"))
  }

  def createPipeline(df: DataFrame, config: Config): PipelineModel = {
    val documentAssembler = new DocumentAssembler().setInputCol(s"text:${config.language}").setOutputCol("document")
    val tokenEmbeddings = config.modelType match {
      case "b" => BertSentenceEmbeddings.pretrained("sent_bert_multi_cased", "xx")
        .setInputCols("document").setOutputCol("embeddings")
      case "u" => UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx")
        .setInputCols("document").setOutputCol("embeddings")
      case _ => XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_base", "xx")
        .setInputCols("document").setOutputCol("embeddings")
    }
    val validPath = s"dat/med/${config.language}"
    if (!Files.exists(Paths.get(validPath))) {
      val preprocessor = new Pipeline().setStages(Array(documentAssembler, tokenEmbeddings))
      val preprocessorModel = preprocessor.fit(df)
      val ef = preprocessorModel.transform(df)
      ef.write.parquet(validPath)
    }
    val classifier = new MultiClassifierDLApproach().setInputCols("embeddings").setOutputCol("category").setLabelColumn(s"ys:${config.language}")
      .setBatchSize(config.batchSize).setMaxEpochs(config.epochs).setLr(config.learningRate.toFloat)
      .setThreshold(config.threshold)
      .setValidationSplit(0.1f)
      .setTestDataset(validPath)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenEmbeddings, classifier))
    pipeline.fit(df)
  }

  def createPipelineMulti(df: DataFrame, config: Config): PipelineModel = {
    val languages = Array("en", "fr", "de", "ja")
    val documentAssemblers = languages.map(lang => new DocumentAssembler().setInputCol(s"text:$lang").setOutputCol(s"document:$lang"))
    val tokenEmbeddings = config.modelType match {
      case "b" => languages.map(lang => BertSentenceEmbeddings.pretrained("sent_bert_multi_cased", "xx")
        .setInputCols(s"document:$lang").setOutputCol(s"embeddings:$lang"))
      case "u" => languages.map(lang => UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx")
        .setInputCols(s"document:$lang").setOutputCol(s"embeddings:$lang"))
      case _ => languages.map(lang => XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_base", "xx")
        .setInputCols(s"document:$lang").setOutputCol(s"embeddings:$lang"))
    }
    val preprocessor = new Pipeline().setStages(documentAssemblers ++ tokenEmbeddings)
    val preprocessorModel = preprocessor.fit(df)
    // compute the average embeddings of the four languages
    val average = udf((e: Array[Float], f: Array[Float], d: Array[Float], j: Array[Float]) => {
      val result = Array.fill[Float](e.size)(0f)
      for (k <- 0 until e.size) result(k) = (e(k) + f(k) + d(k) + j(k)) / 4
      result
    })
    val ef = preprocessorModel.transform(df).withColumn("embeddings",
      average(col("embeddings:en"), col("embeddings:fr"), col("embeddings:de"), col("embeddings:ja"))
    )
    val validPath = s"dat/med/4"
    if (!Files.exists(Paths.get(validPath))) {
      ef.write.parquet(validPath)
    }
    val classifier = new MultiClassifierDLApproach().setInputCols("embeddings").setOutputCol("category").setLabelColumn("ys:en")
      .setBatchSize(config.batchSize).setMaxEpochs(config.epochs).setLr(config.learningRate.toFloat)
      .setThreshold(config.threshold)
      .setValidationSplit(0.1f)
      .setTestDataset(validPath)
    val pipeline = new Pipeline().setStages(documentAssemblers ++ tokenEmbeddings ++ Array(classifier))
    pipeline.fit(df)
  }

  def evaluate(result: DataFrame, config: Config, split: String): Score = {
    val predictionsAndLabels = result.rdd.map { row =>
      (row.getAs[Seq[Double]](0).toArray, row.getAs[Seq[Double]](1).toArray)
    }
    val metrics = new MultilabelMetrics(predictionsAndLabels)
    val ls = metrics.labels
    println("List of all labels: " + ls.mkString(", "))
    val numLabels = if (ls.isEmpty) 0 else (ls.max.toInt + 1) // [0, 1, ..., 22] => 23 positions; [1, 2,...,22] => 23 positions
    val precisionByLabel = Array.fill(numLabels)(0d)
    val recallByLabel = Array.fill(numLabels)(0d)
    val fMeasureByLabel = Array.fill(numLabels)(0d)
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
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("embedding model type, either b/u/x")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('n', "fraction").action((x, conf) => conf.copy(fraction = x)).text("percentage of the dataset to use")
      opt[Double]('e', "threshold").action((x, conf) => conf.copy(threshold = x.toFloat)).text("the minimum threshold for each label to be accepted")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val spark = SparkSession.builder().config("spark.driver.memory", config.driverMemory).master(config.master).appName("MED").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        implicit val formats: AnyRef with Formats = Serialization.formats(NoTypeHints)
        println(Serialization.writePretty(config))
        config.mode match {
          case "train" =>
            val df = readCorpus(spark, config.language).sample(config.fraction, 220712L)
              .filter(col(s"ys:${config.language}") =!= Array("0"))
            df.show()
            df.printSchema()
            println (s"Number of samples = ${df.count}")
            val model = createPipeline(df, config)
            val ef = model.transform(df)
            ef.show()
            // convert the "category" column (of type Array[String]) to the the "prediction" column of type List[Double] for evaluation
            val ff = ef.mapAnnotationsCol("category", "prediction", "category", (a: Seq[Annotation]) => if (a.nonEmpty) a.map (_.result.toDouble) else List.empty[Double])
              // convert the "ys" column (of type Array[String]) to the  "label" column of type List[Double] for evaluation
            val f = udf((ys: List[String]) => ys.map (_.toDouble))
            val gf = ff.withColumn("label", f(col(s"ys:${config.language}")))
            gf.show()
            gf.printSchema()
            val score = evaluate(gf.select("prediction", "label"), config, "train")
            println(Serialization.writePretty(score))
          case "trainMulti" =>
            val df = readCorpus(spark)
              .filter(col(s"ys:${config.language}") =!= Array("0"))
            println("Number of samples = " + df.count)
            df.printSchema()
            df.select("id", "text:en", "text:fr", "text:de", "text:ja", "ys:en", "ys:fr", "ys:de", "ys:ja").show()
            val model = createPipelineMulti(df, config)
            val ef = model.transform(df)
            // convert the "category" column (of type Array[String]) to the the "prediction" column of type List[Double] for evaluation
            val ff = ef.mapAnnotationsCol("category", "prediction", "category", (a: Seq[Annotation]) => if (a.nonEmpty) a.map(_.result.toDouble) else List.empty[Double])
            // convert the "ys" column (of type Array[String]) to the  "label" column of type List[Double] for evaluation
            val f = udf((ys: List[String]) => ys.map(_.toDouble))
            val gf = ff.withColumn("label", f(col(s"ys:${config.language}")))
            gf.show()
            gf.printSchema()
            val score = evaluate(gf.select("prediction", "label"), config, "train")
            println(Serialization.writePretty(score))
          case "eval" =>
          case _ =>
        }
        spark.stop()
      case None =>
    }
  }
}
