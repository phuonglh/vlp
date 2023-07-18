package vlp.med

import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler}
import com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach
import com.johnsnowlabs.nlp.embeddings.{BertSentenceEmbeddings, DeBertaEmbeddings, SentenceEmbeddings, UniversalSentenceEncoder, XlmRoBertaEmbeddings, XlmRoBertaSentenceEmbeddings}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions.{col, flatten, length, lit, not, size, trim, udf}
import org.json4s.{Formats, NoTypeHints}
import org.json4s.jackson.Serialization
import scopt.OptionParser

import java.nio.file.{Files, Paths}
import com.johnsnowlabs.nlp.functions._
import com.intel.analytics.bigdl.dllib.keras.layers.Dense
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.models.{KerasNet, Models}
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.keras.metrics.CategoricalAccuracy
import com.intel.analytics.bigdl.dllib.optim.{MAE, Trigger}
import com.intel.analytics.bigdl.dllib.utils.{Engine, Shape}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.nnframes.{NNEstimator, NNModel}
import com.intel.analytics.bigdl.dllib.nn.BCECriterion
import com.johnsnowlabs.nlp.base.EmbeddingsFinisher
import com.johnsnowlabs.nlp.annotators.Tokenizer
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, StopWordsRemover, VectorAssembler, Tokenizer => TokenizerSpark}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{DenseVector, Vectors}

import java.io.PrintWriter
import java.io.File


object MED {
  private val headerMap = scala.collection.mutable.Map[String, String]()

  def readCorpusMono(spark: SparkSession, language: String, split: String): DataFrame = {
    val path = if (split == "train") {
      s"dat/med/ntcir17_mednlp-sc_sm_train_26_06_23/${language}.csv"
    } else {
      s"dat/med/ntcir17_mednlp-sc_sm_test_10_07_23_unlabeled/${language}.csv"
    }
    import spark.implicits._
    val df = spark.read.text(path).filter(length(trim(col("value"))) > 0)
    val header = df.first().get(0)
    headerMap += (language -> header.toString)
    df.filter(not(col("value").contains(header))).map { row =>
      val line = row.getAs[String](0)
      val j = line.indexOf(",")
      if (j <= 1 || j > 8) println("ERROR: " + line)
      val n = line.length
      val trainId = line.substring(0, j).toDouble
      var text = line.substring(j+1, n-43)
      // remove quotes at the beginning or the end of the text if there are
      if (text.startsWith("\"")) text = text.substring(1, text.length - 2) // account for 1 comma
      val ys = line.substring(n-43) // 22 binary labels, plus 21 commas
      // convert to 23 labels [0, 1, 2,..., 22]. The zero vector corresponds to "0". Other labels are shifted by their index by 1.
      val labels = ys.split(",").zipWithIndex.filter(_._1 == "1").map(p => (p._2 + 1).toString)
      (trainId, text, if (labels.length > 0) labels else Array("0"))
    }.toDF("id", s"text:${language}", s"ys:${language}")
  }

  def readCorpusMany(spark: SparkSession, language: String, split: String): DataFrame = {
    val path = if (split == "train") {
      s"dat/med/ntcir17_mednlp-sc_sm_train_26_06_23/${language}.csv"
    } else {
      s"dat/med/ntcir17_mednlp-sc_sm_test_10_07_23_unlabeled/${language}.csv"
    }
    import spark.implicits._
    val df = spark.read.text(path).filter(length(trim(col("value"))) > 0)
    val header = df.first().get(0)
    headerMap += (language -> header.toString)
    df.filter(not(col("value").contains(header))).map { row =>
      val line = row.getAs[String](0)
      val j = line.indexOf(",")
      if (j <= 1 || j > 8) println("ERROR: " + line)
      val n = line.length
      val trainId = line.substring(0, j).toDouble
      var text = line.substring(j + 1, n - 43)
      // remove quotes at the beginning or the end of the text if there are
      if (text.startsWith("\"")) text = text.substring(1, text.length - 2)
      val ys = line.substring(n - 43) // 22 binary labels, plus 21 commas
      // convert 23 labels into a multi-hot vector
      val labels = Vectors.dense(ys.split(",").map(_.toDouble))
      (trainId, text, labels)
    }.toDF("id", s"text:${language}", s"ys:${language}")
  }


  def readCorpus(spark: SparkSession, sparseLabel: Boolean, split: String): DataFrame = {
    val languages = List("de", "en", "fr", "ja")
    val dfs = languages.map { lang =>
      if (sparseLabel) readCorpusMono(spark, lang, split) else readCorpusMany(spark, lang, split)
    }
    dfs.reduce((df1, df2) => df1.join(df2, "id"))
  }

  def createPipelineMono(df: DataFrame, config: Config): PipelineModel = {
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
      .setThreshold(config.threshold.toFloat)
      .setTestDataset(validPath)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenEmbeddings, classifier))
    pipeline.fit(df)
  }

  private def exportEmbeddings(df: DataFrame, modelType: String, lang: String, outputPath: String): Unit = {
    val documentAssembler = new DocumentAssembler().setInputCol(s"text:$lang").setOutputCol(s"document:$lang")
    val tokenEmbeddings = modelType match {
      case "u" => UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx")
        .setInputCols(s"document:$lang").setOutputCol(s"embeddings:$lang")
      case "b" => BertSentenceEmbeddings.pretrained("sent_bert_multi_cased", "xx")
        .setInputCols(s"document:$lang").setOutputCol(s"embeddings:$lang")
      case "d" => DeBertaEmbeddings.pretrained("mdeberta_v3_base", "xx")
        .setInputCols (s"document:$lang", s"token:$lang").setOutputCol (s"token:embeddings:$lang")
      case "r" => XlmRoBertaEmbeddings.pretrained("xlm_roberta_large", "xx")
        .setInputCols (s"document:$lang", s"token:$lang").setOutputCol (s"token:embeddings:$lang")
      case _ => XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_base", "xx")
        .setInputCols(s"document:$lang").setOutputCol(s"embeddings:$lang")
    }
    val embeddingsFinisher = new EmbeddingsFinisher().setInputCols(s"embeddings:$lang").setOutputCols(s"$lang")
    val pipeline = if (Set("d", "r").contains(modelType)) {
      val tokenizer = new Tokenizer().setInputCols(Array(s"document:$lang")).setOutputCol(s"token:$lang")
      val sentenceEmbedding = new SentenceEmbeddings().setInputCols(s"document:$lang", s"token:embeddings:$lang").setOutputCol(s"embeddings:$lang")
      new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenEmbeddings, sentenceEmbedding, embeddingsFinisher))
    } else {
      new Pipeline().setStages(Array(documentAssembler, tokenEmbeddings, embeddingsFinisher))
    }
    val model = pipeline.fit(df)
    val outputCol = lang.substring(0, 1)
    val ef = model.transform(df)
      .withColumn(outputCol, flatten(col(s"$lang")))
      .select("id", outputCol, s"ys:$lang")
    ef.write.mode(SaveMode.Overwrite).parquet(s"$outputPath/$lang/$modelType")
  }

  // the MultiClassifierDLApproach of Spark-NLP expects a SentenceEmbedding as input column.
  // Here, we use a simple vector as feature, so we develop a BigDL model to plug-in
  private def createModel(featureSize: Int, hiddenSize: Int, labelSize: Int): KerasNet[Float] = {
    val model = Sequential()
    model.add(Dense(outputDim = hiddenSize, inputShape = Shape(featureSize), activation = "relu").setName(s"Dense"))
    // add the last layer for BCE loss
    // sigmoid for multi-label instead of softmax, which gives better performance
    model.add(Dense(labelSize, activation = "sigmoid").setName("Dense-output"))
    model
  }

  def createPipelineMany(df: DataFrame, config: Config): NNModel[Float] = {
    var featureSize: Int = config.modelType match {
      case "u" => 512
      case "r" => 1024
      case _ => 768
    }
    if (config.concat) featureSize = featureSize * 4

    val bigdl = createModel(featureSize, config.hiddenSize, 22)
    val estimator = NNEstimator(bigdl, BCECriterion(), Array(featureSize), Array(22))
    val trainingSummary = TrainSummary(appName = config.modelType, logDir = s"sum/med/")
    val validationSummary = ValidationSummary(appName = config.modelType, logDir = s"sum/med/")

    estimator.setLabelCol("ys:en").setFeaturesCol("features")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(config.learningRate))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, df, Array(new CategoricalAccuracy(), new MAE()), config.batchSize)
    val model = estimator.fit(df)
    bigdl.saveModel(s"dat/out/${config.modelType}.bigdl", overWrite = true)
    model
  }

  def createPipelineDiscrete(df: DataFrame, config: Config): (PipelineModel, NNModel[Float]) = {
    // English
    val tokenizerEn = new TokenizerSpark().setInputCol("text:en").setOutputCol("token:en")
    val stopWordsRemoverEn = new StopWordsRemover().setInputCol("token:en").setOutputCol("word:en")
    val countVectorizerEn = new CountVectorizer().setInputCol("word:en").setOutputCol("e").setBinary(true).setMinDF(2)
    // French
    val tokenizerFr = new TokenizerSpark().setInputCol("text:fr").setOutputCol("token:fr")
    val stopWordsRemoverFr = new StopWordsRemover().setInputCol("token:fr").setOutputCol("word:fr")
      .setStopWords(StopWordsRemover.loadDefaultStopWords("french"))
    val countVectorizerFr = new CountVectorizer().setInputCol("word:fr").setOutputCol("f").setBinary(true).setMinDF(2)
    // German
    val tokenizerDe = new TokenizerSpark().setInputCol("text:de").setOutputCol("token:de")
    val stopWordsRemoverDe = new StopWordsRemover().setInputCol("token:de").setOutputCol("word:de")
      .setStopWords(StopWordsRemover.loadDefaultStopWords("german"))
    val countVectorizerDe = new CountVectorizer().setInputCol("word:de").setOutputCol("d").setBinary(true).setMinDF(2)
    //
    val assembler = new VectorAssembler().setInputCols(Array("e", "f", "d")).setOutputCol("features")
    val preprocessorPipeline = new Pipeline().setStages(Array(
      tokenizerEn, stopWordsRemoverEn, countVectorizerEn,
      tokenizerFr, stopWordsRemoverFr, countVectorizerFr,
      tokenizerDe, stopWordsRemoverDe, countVectorizerDe,
      assembler
    ))
    val preprocessor = preprocessorPipeline.fit(df)
    val featureSize = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary.size +
      preprocessor.stages(5).asInstanceOf[CountVectorizerModel].vocabulary.size +
      preprocessor.stages(8).asInstanceOf[CountVectorizerModel].vocabulary.size
    println(s"featureSize = $featureSize")

    val ef = preprocessor.transform(df)
    val bigdl = createModel(featureSize, config.hiddenSize, 22)
    val estimator = NNEstimator(bigdl, BCECriterion(), Array(featureSize), Array(22))
    val trainingSummary = TrainSummary(appName = "discrete", logDir = s"sum/med/")
    val validationSummary = ValidationSummary(appName = "discrete", logDir = s"sum/med/")

    estimator.setLabelCol("ys:en").setFeaturesCol("features")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(config.learningRate))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, ef, Array(new CategoricalAccuracy(), new MAE()), config.batchSize)
    val model = estimator.fit(ef)
    bigdl.saveModel(s"dat/out/discrete.bigdl", overWrite = true)
    (preprocessor, model)
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

  def predict(df: DataFrame, bigdl: KerasNet[Float]): DataFrame = {
    println(bigdl.summary())
    val model = NNModel(bigdl)
    model.transform(df)
  }

  // Option 1: compute the average embeddings of the four languages
  val average = udf((e: Array[Float], f: Array[Float], d: Array[Float], j: Array[Float]) => {
    val result = Array.fill[Float](e.length)(0f)
    for (k <- e.indices)
      result(k) = (e(k) + f(k) + d(k) + j(k)) / 4
    result
  })
  // Option 2: concatenate the embeddings of the four language
  val concatenate = udf((e: Array[Float], f: Array[Float], d: Array[Float], j: Array[Float]) => {
    e ++ f ++ d ++ j
  })

  // for evaluation using Spark
  val sparsifyVector = udf((target: DenseVector) => {
    val x = target.toArray.zipWithIndex.filter(_._1 == 1.0).map(_._2.toDouble)
    if (x.isEmpty) Array(0d) else x
  })
  val sparsifyArray = udf((target: Array[Float], threshold: Double) => {
    val x = target.zipWithIndex.filter(_._1 >= threshold).map(_._2.toDouble)
    if (x.isEmpty) Array(0d) else x
  })

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
      opt[Double]('e', "threshold").action((x, conf) => conf.copy(threshold = x)).text("the minimum threshold for each label to be accepted")
      opt[Int]('h', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("hidden size")
      opt[Boolean]('c', "concatenation").action((x, conf) => conf.copy(concat = true)).text("concatenate embedding vectors, default is false")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val conf = Engine.createSparkConf().setAppName(getClass.getName).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        Engine.init
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        implicit val formats: AnyRef with Formats = Serialization.formats(NoTypeHints)
        println(Serialization.writePretty(config))
        config.mode match {
          case "trainMono" =>
            val df = readCorpusMono(spark, config.language, "train").sample(config.fraction, 220712L)
//              .filter(col(s"ys:${config.language}") =!= Array("0"))
            println(headerMap)
            df.show()
            df.printSchema()
            println (s"Number of samples = ${df.count}")
            val model = createPipelineMono(df, config)
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
          case "trainMany" =>
            val languages = Array("en", "de", "fr", "ja")
            val dfs = languages.map(lang => spark.read.parquet(s"dat/med/train/$lang/${config.modelType}"))
            val df = dfs.reduce((df1, df2) => df1.join(df2, "id"))
            val ef = if (config.concat) {
              df.withColumn("features", concatenate(col("e"), col("d"), col("f"), col("j")))
            } else {
              df.withColumn("features", average(col("e"), col("d"), col("f"), col("j")))
            }.select("id", "features", "ys:en")
            ef.printSchema()
            ef.withColumn("size", size(col("features"))).show()

            val model = createPipelineMany(ef, config)
            val ff = model.transform(ef)
            ff.show()
            ff.printSchema()
            val gf = ff.withColumn("target", sparsifyVector(col("ys:en")))
              .withColumn("output", sparsifyArray(col("prediction"), lit(config.threshold)))
            val score = evaluate(gf.select("output", "target"), config, "train")
            println(Serialization.writePretty(score))
            // save result to CSV file for submission
            val binarize = udf((prediction: Array[Float]) =>
              prediction.map(e => if (e >= config.threshold) 1 else 0).mkString(",")
            )
            val output = ff.withColumn("output", binarize(col("prediction")))
            output.printSchema()
            languages.foreach { lang =>
              val sf = readCorpusMono(spark, lang, "train").select("id", s"text:$lang")
              val uf = sf.join(output, "id")
              val result = uf.select("id", s"text:$lang", "output").repartition(1)
              val lines = result.collect().map { row =>
                val id = row.getAs[Double](0)
                val text = row.getAs[String](1)
                val labels = row.getAs[String](2)
                id + ",\"" + text + "\"," + labels
              }
              val pathOut = s"dat/out/${config.modelType}/ntcir17_mednlp-sc_sm_${lang}_train_1.csv"
              val writer = new PrintWriter(new File(pathOut))
              writer.println(headerMap(s"$lang"))
              lines.foreach(line => writer.println(line))
              writer.close()
            }
          case "eval" =>
          case "exportEmbeddingTrain" =>
            val df = readCorpus(spark, sparseLabel = false, "train")
            val languages = Array("en", "de", "fr", "ja")
            languages.foreach(exportEmbeddings(df, config.modelType, _, "dat/med/train"))
          case "exportEmbeddingTest" =>
            val df = readCorpus(spark, sparseLabel = false, "test")
            df.show()
            df.count()
            val languages = Array("en", "fr", "de", "ja")
            languages.foreach(exportEmbeddings(df, config.modelType, _, "dat/med/test"))
          case "submit" =>
            // perform training and prediction at the same time (the BigDL load method has a bug!)
            val languages = Array("en", "de", "fr", "ja")
            val dfs = languages.map(lang => spark.read.parquet(s"dat/med/train/$lang/${config.modelType}"))
            val df = dfs.reduce((df1, df2) => df1.join(df2, "id"))
            val dfsTest = languages.map(lang => spark.read.parquet(s"dat/med/test/$lang/${config.modelType}"))
            val dfTest = dfsTest.reduce((df1, df2) => df1.join(df2, "id"))
            val ef = if (config.concat) {
              df.withColumn("features", concatenate(col("e"), col("d"), col("f"), col("j")))
            } else {
              df.withColumn("features", average(col("e"), col("d"), col("f"), col("j")))
            }.select("id", "features", "ys:en")
            ef.printSchema()
            ef.withColumn("size", size(col("features"))).show()
            val efTest = if (config.concat) {
              dfTest.withColumn("features", concatenate(col("e"), col("d"), col("f"), col("j")))
            } else {
              dfTest.withColumn("features", average(col("e"), col("d"), col("f"), col("j")))
            }.select("id", "features", "ys:en")

            val model = createPipelineMany(ef, config)
            val ff = model.transform(ef)
            ff.show()
            ff.printSchema()
            val ffTest = model.transform(efTest)
            val gf = ff.withColumn("target", sparsifyVector(col("ys:en")))
              .withColumn("output", sparsifyArray(col("prediction"), lit(config.threshold)))
            val score = evaluate(gf.select("output", "target"), config, "train")
            println(Serialization.writePretty(score))
            // save result to CSV file for submission
            val binarize = udf((prediction: Array[Float]) =>
              prediction.map(e => if (e >= config.threshold) 1 else 0).mkString(",")
            )
            val output = ff.withColumn("output", binarize(col("prediction")))
            output.printSchema()
            val outputTest = ffTest.withColumn("output", binarize(col("prediction")))
            languages.foreach { lang =>
              // write train result
              val sf = readCorpusMono(spark, lang, "train").select("id", s"text:$lang")
              val uf = sf.join(output, "id")
              val result = uf.select("id", s"text:$lang", "output").repartition(1)
              val lines = result.collect().map { row =>
                val id = row.getAs[Double](0)
                val text = row.getAs[String](1)
                val labels = row.getAs[String](2)
                id + ",\"" + text + "\"," + labels
              }
              val pathOut = s"dat/out/${config.modelType}/ntcir17_mednlp-sc_sm_${lang}_train_1.csv"
              val writer = new PrintWriter(new File(pathOut))
              writer.println(headerMap(s"$lang"))
              lines.foreach(line => writer.println(line))
              writer.close()
              // write test result
              val sfTest = readCorpusMono(spark, lang, "test").select("id", s"text:$lang")
              val ufTest = sfTest.join(outputTest, "id")
              val resultTest = ufTest.select("id", s"text:$lang", "output").repartition(1)
              val linesTest = resultTest.collect().map { row =>
                val id = row.getAs[Double](0)
                val text = row.getAs[String](1)
                val labels = row.getAs[String](2)
                id + ",\"" + text + "\"," + labels
              }
              val pathOutTest = s"dat/out/${config.modelType}/ntcir17_mednlp-sc_sm_${lang}_test_1.csv"
              val writerTest = new PrintWriter(new File(pathOutTest))
              writerTest.println(headerMap(s"$lang"))
              linesTest.foreach(line => writerTest.println(line))
              writerTest.close()
            }
          case "trainDiscrete" =>
            val df = readCorpus(spark, false, "train")
            val (preprocessor, model) = createPipelineDiscrete(df, config)
            val ef = preprocessor.transform(df)
            val ff = model.transform(ef)
            val gf = ff.withColumn("target", sparsifyVector(col("ys:en")))
              .withColumn("output", sparsifyArray(col("prediction"), lit(config.threshold)))
            val score = evaluate(gf.select("output", "target"), config, "train")
            println(Serialization.writePretty(score))
            ff.show()
            val dfTest = readCorpus(spark, false, "test")
            val efTest = preprocessor.transform(dfTest)
            val binarize = udf((prediction: Array[Float]) =>
              prediction.map(e => if (e >= config.threshold) 1 else 0).mkString(",")
            )
            val ffTest = model.transform(efTest).withColumn("output", binarize(col("prediction")))

            val languages = Array("en", "fr", "de", "ja")
            languages.foreach { lang =>
              val sfTest = readCorpusMono(spark, lang, "test").select("id")
              val ufTest = sfTest.join(ffTest, "id")
              ufTest.show()
              // write test result
              val resultTest = ufTest.select("id", s"text:$lang", "output").repartition(1)
              val linesTest = resultTest.collect().map { row =>
                val id = row.getAs[Double](0)
                val text = row.getAs[String](1)
                val labels = row.getAs[String](2)
                id + ",\"" + text + "\"," + labels
              }
              val pathOutTest = s"dat/out/discrete/ntcir17_mednlp-sc_sm_${lang}_test_1.csv"
              val writerTest = new PrintWriter(new File(pathOutTest))
              writerTest.println(headerMap(s"$lang"))
              linesTest.foreach(line => writerTest.println(line))
              writerTest.close()
            }
          case _ =>
        }
        spark.stop()
      case None =>
    }
  }
}
