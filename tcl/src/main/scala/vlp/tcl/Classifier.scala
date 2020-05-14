package vlp.tcl

import java.io.{FileOutputStream, OutputStreamWriter, PrintWriter}
import java.nio.charset.StandardCharsets

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Dataset, DataFrame, RowFactory, SparkSession}
import org.apache.spark.sql.types.{DataType, StringType, StructField, StructType}
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import org.slf4j.LoggerFactory
import scopt.OptionParser

/**
  * phuonglh, 5/28/18, 1:01 PM
  */
class TCL(val sparkContext: SparkContext, val config: ConfigTCL) {
  final val logger = LoggerFactory.getLogger(getClass.getName)
  var numCats: Int = 0
  val sparkSession = SparkSession.builder().getOrCreate()
  import sparkSession.implicits._

  def createDataset: Dataset[Document] = {
    val data = TCL.vnexpress(sparkSession, config.data).as[Document]
    numCats = data.select("category").distinct().count().toInt
    logger.info("#(categories) = " + numCats)
    val g = data.groupBy("category").count()
    g.show()
    data
  }

  def train(dataset: Dataset[Document]): PipelineModel = {
    dataset.cache()
    // create pipeline
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    val stopWordsRemover = new StopWordsRemover().setInputCol("tokens").setOutputCol("unigrams").setStopWords(StopWords.punctuations)
    val unigramCounter = new CountVectorizer().setInputCol("unigrams").setOutputCol("us").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)
    val labelIndexer = new StringIndexer().setInputCol("category").setHandleInvalid("keep").setOutputCol("label")
    val pipeline = if (config.classifier == "mlr") {
      val bigram = new NGram().setInputCol("unigrams").setOutputCol("bigrams").setN(2)
      val bigramCounter = new CountVectorizer().setInputCol("bigrams").setOutputCol("bs").setMinDF(config.minFrequency).setVocabSize(2*config.numFeatures)
      val assembler = new VectorAssembler().setInputCols(Array("us", "bs")).setOutputCol("features")
      val mlr = new LogisticRegression().setMaxIter(config.iterations).setRegParam(config.lambda).setStandardization(false)
      new Pipeline().setStages(Array(labelIndexer, tokenizer, stopWordsRemover, unigramCounter, bigram, bigramCounter, assembler, mlr))
    } else {
      val featureHashing = new HashingTF().setInputCol("unigrams").setOutputCol("features").setNumFeatures(config.numFeatures).setBinary(true)
      val xs = config.layers.trim
      val hiddenLayers = if (xs.nonEmpty) xs.split("[\\s,]+").map(_.toInt); else Array[Int]()
      val layers = Array(config.numFeatures) ++ hiddenLayers ++ Array[Int](numCats)
      logger.info(layers.mkString(", "))
      val mlp = new MultilayerPerceptronClassifier().setMaxIter(config.iterations).setBlockSize(config.batchSize).setSeed(124456).setLayers(layers)
      new Pipeline().setStages(Array(labelIndexer, tokenizer, stopWordsRemover, featureHashing, mlp))
    }
    logger.info("#(documents) = " + dataset.count())
    logger.info("Training process started. Please wait...")
    val model = pipeline.fit(dataset)
    model.write.overwrite().save(config.modelPath)
    model
  }

  def eval(dataset: Dataset[Document]): Unit = {
    val model = PipelineModel.load(config.modelPath)

    val transformer = model.stages(3)
    if (transformer.isInstanceOf[CountVectorizerModel]) {
      val vocabulary = transformer.asInstanceOf[CountVectorizerModel].vocabulary
      logger.info("#(vocabulary) = " + vocabulary.size)
    } else if (transformer.isInstanceOf[HashingTF]) {
        val numFeatures = transformer.asInstanceOf[HashingTF].getNumFeatures
        logger.info("#(numFeatures) = " + numFeatures)
    } else logger.error(s"Error in reading information from ${transformer.getClass.getName}")

    val outputDF = model.transform(dataset)

    import sparkSession.implicits._
    val predictionAndLabels = outputDF.select("label", "prediction").map(row => (row.getDouble(0), row.getDouble(1))).rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val scores = (metrics.accuracy, metrics.weightedFMeasure)
    logger.info(s"'prediction' scores = $scores")
    if (config.verbose) {
      outputDF.show(10)
      val labels = metrics.labels
      labels.foreach(label => {
        val sb = new StringBuilder()
        sb.append(s"Precision($label) = " + metrics.precision(label) + ", ")
        sb.append(s"Recall($label) = " + metrics.recall(label) + ", ")
        sb.append(s"F($label) = " + metrics.fMeasure(label))
        logger.info(sb.toString)
      })
    }
  }

  def test(dataset: Dataset[Document], outputFile: String): Unit = {
    val model = PipelineModel.load(config.modelPath)
    val outputDF = model.transform(dataset)
    import sparkSession.implicits._
    val prediction = outputDF.select("category", "content", "prediction", "probability")
      .map(row => (row.getString(0), row.getString(1), row.getDouble(2).toInt, row.getAs[DenseVector](3)))
      .collect()
    val writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(outputFile), StandardCharsets.UTF_8), true)
    val labels = model.stages(0).asInstanceOf[StringIndexerModel].labels
    prediction.foreach(pair => {
      writer.print(pair._1)
      writer.print("\t")
      writer.print(pair._2)
      writer.print("\t")
      writer.print(labels(pair._3))
      writer.print("\t")
      writer.println(pair._4.toArray(pair._3))
    })
    writer.close()
  }

  /**
    * Predicts a document and outputs a probability distribution of
    * categories.
    * @param document a document
    * @param model the pipeline model
    * @return a probability distribution
    */
  def predict(document: Document, model: PipelineModel): Map[String, Double] = {
    val xs = List(document)
    val labels = model.stages(0).asInstanceOf[StringIndexerModel].labels
    import sparkSession.implicits._
    val dataset = sparkSession.createDataset(xs).as[Document]
    val outputDF = model.transform(dataset)
    outputDF.select("probability")
      .map(row => row.getAs[DenseVector](0).values).head()
      .zipWithIndex
      .map(pair => (if (pair._2 < labels.size) labels(pair._2) else "NA", pair._1)).toMap
  }

  def predict(inputFile: String, outputFile: String): Unit = {
    val lines = scala.io.Source.fromFile(inputFile)("UTF-8").getLines.toList.filter(_.trim.nonEmpty)
    val xs = lines.map { line =>
      val j = line.indexOf(' ')
      val y = line.substring(0, j)
      val x = line.substring(j+1)
      val tokens = vlp.tok.Tokenizer.tokenize(x).map(_._3)
      Document(y, tokens.mkString(" "))
    }
    import sparkSession.implicits._
    val data = sparkSession.createDataset(xs).as[Document]
    test(data, outputFile)
  }
}

object TCL {
  final val logger = LoggerFactory.getLogger(getClass.getName)


  def vnexpress(sparkSession: SparkSession, path: String): DataFrame = {
    val rdd = sparkSession.sparkContext.textFile(path).map(_.trim).filter(_.nonEmpty)
    val rows = rdd.map { line =>
      val parts = line.split("\\t+")
      RowFactory.create(parts(0).trim, parts(1).trim)
    }
    val schema = StructType(Array(StructField("category", StringType, false), StructField("text", StringType, false)))
    sparkSession.createDataFrame(rows, schema)
  }

  /**
    * Samples a percentage of an input data and write that sample into an output path.
    * @param sparkSession
    * @param inputPath
    * @param outputPath
    * @param percentage
    */
  def sampling(sparkSession: SparkSession, inputPath: String, outputPath: String, percentage: Double = 0.1): Unit = {
    val rdd = sparkSession.sparkContext.textFile(inputPath).map(_.trim).filter(_.nonEmpty)
    val sample = rdd.sample(false, percentage, 220712L)
    sample.repartition(10).saveAsTextFile(outputPath)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val parser = new OptionParser[ConfigTCL]("vlp.tcl") {
      head("vlp.tcl", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('h', "hiddenUnits").action((x, conf) => conf.copy(hiddenUnits = x)).text("number of hidden units in each layer")
      opt[String]('l', "layers").action((x, conf) => conf.copy(layers = x)).text("layers config of MLP")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[String]('d', "data").action((x, conf) => conf.copy(data = x)).text("data path, default is 'dat/fin/*.txt'")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/tcl/'")
      opt[String]('i', "input").action((x, conf) => conf.copy(input = x)).text("input path")
      opt[String]('o', "output").action((x, conf) => conf.copy(output = x)).text("output path")
    }
    parser.parse(args, ConfigTCL()) match {
      case Some(config) =>
        val sparkSession = SparkSession.builder().appName(getClass.getName).master(config.master).getOrCreate()
        implicit val formats = Serialization.formats(NoTypeHints)
        logger.info(Serialization.writePretty(config))
        val tcl = new TCL(sparkSession.sparkContext, config)
        val dataset = tcl.createDataset
        val Array(training, test) = dataset.randomSplit(Array(0.8, 0.2), seed = 20150909)
        config.mode match {
          case "train" => {
            training.show()
            tcl.train(training)
            tcl.eval(training)
            test.show()
            tcl.eval(test)
          }
          case "eval" => {
            test.show()
            tcl.eval(test)
          }
          case "test" => tcl.test(test, config.output)
          case "sample" => sampling(sparkSession, "dat/vne/5cats.txt", "dat/vne/5catsSample", 0.01)
          case "predict" => {
            val document = Document("Other", "CTCP Nước và Môi trường Bình Dương (Biwase – BWE) đã sớm chốt danh sách tham dự kỳ họp ĐHCĐ thường niên năm 2019 từ cuối năm 2018, ngày 28/12/2018 kèm với đó là việc chi tạm ứng cổ tức bằng tiền tỷ lệ 7%. Mới đây Biwase đã phát đi thông báo mời họp ĐHCĐ thường niên vào 14h ngày 15/3/2019 tại văn phòng công ty.")
            val model = PipelineModel.load(config.modelPath)
            val prediction = tcl.predict(document, model)
            logger.info(prediction.toString)
          }
        }
        sparkSession.stop()
      case None =>
      }
  }
}
