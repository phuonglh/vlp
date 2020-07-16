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
import vlp.tok.SentenceDetection
import org.apache.spark.ml.classification.RandomForestClassifier
import java.io.File

/**
  * phuonglh, 5/28/18, 1:01 PM
  */
class Classifier(val sparkContext: SparkContext, val config: ConfigTCL) {
  final val logger = LoggerFactory.getLogger(getClass.getName)
  val sparkSession = SparkSession.builder().getOrCreate()
  import sparkSession.implicits._

  def createDataset(path: String, numberOfSentences: Int = Int.MaxValue): Dataset[Document] = {
    Classifier.readTextData(sparkSession, path, numberOfSentences).as[Document]
  }

  def train(dataset: Dataset[Document]): PipelineModel = {
    dataset.cache()
    // create pipeline
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    val stopWordsRemover = new StopWordsRemover().setInputCol("tokens").setOutputCol("unigrams").setStopWords(StopWords.punctuations)
    val unigramCounter = new CountVectorizer().setInputCol("unigrams").setOutputCol("us").setMinDF(config.minFrequency).setVocabSize(config.numFeatures)
    val labelIndexer = new StringIndexer().setInputCol("category").setHandleInvalid("skip").setOutputCol("label")
    val classifierType = config.classifier
    val pipeline = if (classifierType == "mlr") {
      val bigram = new NGram().setInputCol("unigrams").setOutputCol("bigrams").setN(2)
      val bigramCounter = new CountVectorizer().setInputCol("bigrams").setOutputCol("bs").setMinDF(config.minFrequency).setVocabSize(2*config.numFeatures)
      val assembler = new VectorAssembler().setInputCols(Array("us", "bs")).setOutputCol("features")
      val mlr = new LogisticRegression().setMaxIter(config.iterations).setRegParam(config.lambda).setStandardization(false)
      new Pipeline().setStages(Array(labelIndexer, tokenizer, stopWordsRemover, unigramCounter, bigram, bigramCounter, assembler, mlr))
    } else if (classifierType == "mlp") {
      val featureHashing = new HashingTF().setInputCol("unigrams").setOutputCol("features").setNumFeatures(config.numFeatures).setBinary(true)
      val numLabels = labelIndexer.fit(dataset).labels.size
      logger.info(s"numLabels = ${numLabels}")
      val xs = config.hiddenUnits.trim
      val hiddenLayers = if (xs.nonEmpty) xs.split("[\\s,]+").map(_.toInt); else Array[Int]()
      val layers = Array(config.numFeatures) ++ hiddenLayers ++ Array[Int](numLabels)
      logger.info(layers.mkString(", "))
      val mlp = new MultilayerPerceptronClassifier().setMaxIter(config.iterations).setBlockSize(config.batchSize).setSeed(123).setLayers(layers)
      new Pipeline().setStages(Array(labelIndexer, tokenizer, stopWordsRemover, featureHashing, mlp))
    } else if (classifierType == "rfc") {
      val featureHashing = new HashingTF().setInputCol("unigrams").setOutputCol("features").setNumFeatures(config.numFeatures).setBinary(false)
      val rfc = new RandomForestClassifier().setNumTrees(config.numTrees).setMaxDepth(config.maxDepth)
      new Pipeline().setStages(Array(labelIndexer, tokenizer, stopWordsRemover, featureHashing, rfc))
    } else {
      logger.error("Not support classifier type: " + classifierType)
      new Pipeline()
    }
    logger.info("#(documents) = " + dataset.count())
    logger.info("Training process started. Please wait...")
    val model = pipeline.fit(dataset)
    model.write.overwrite().save(config.modelPath + "/" + classifierType)
    model
  }

  def eval(model: PipelineModel, dataset: Dataset[Document]): Unit = {
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

  def eval(dataset: Dataset[Document]): Unit = {
    val model = PipelineModel.load(config.modelPath + "/" + config.classifier.toLowerCase())
    eval(model, dataset)
  }

  def test(dataset: Dataset[Document], outputFile: String): Unit = {
    val model = PipelineModel.load(config.modelPath + "/" + config.classifier.toLowerCase())
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
      val tokens = vlp.tok.Tokenizer.tokenize(line).map(_._3)
      Document("NA", tokens.mkString(" "))
    }
    import sparkSession.implicits._
    val data = sparkSession.createDataset(xs).as[Document]
    test(data, outputFile)
  }
}

object Classifier {
  final val logger = LoggerFactory.getLogger(getClass.getName)

  /**
    * Loads text data into a data frame. If number of sentences are positive then only that number of sentences 
    * for each document are loaded. Default is a negative value -1, which means all the content will be loaded.
    *
    * @param sparkSession
    * @param path path to the data file(s)
    * @param numberOfSentences
    * @return a data frame of two columns (category, text)
    */
  def readTextData(sparkSession: SparkSession, path: String, numberOfSentences: Int = Int.MaxValue): DataFrame = {
    val rdd = sparkSession.sparkContext.textFile(path).map(_.trim).filter(_.nonEmpty)
    val rows = rdd.map { line =>
      val parts = line.split("\\t+")
      val text = SentenceDetection.run(parts(1).trim, numberOfSentences).mkString(" ")
      RowFactory.create(parts(0).trim, text)
    }
    val schema = StructType(Array(StructField("category", StringType, false), StructField("text", StringType, false)))
    sparkSession.createDataFrame(rows, schema)
  }

  def readSHINRA(sparkSession: SparkSession, path: String, numberOfSentences: Int = Int.MaxValue): Dataset[Document] = {
    val rdd = sparkSession.sparkContext.textFile(path).map(_.trim).filter(_.nonEmpty)
    val rows = rdd.map { line =>
      var p = line.indexOf('\t')
      val q = line.lastIndexOf('\t')
      val text = SentenceDetection.run(line.substring(p+1, q), numberOfSentences).replaceAll("\u200b", "").mkString(" ")
      // use only first category
      val category = line.substring(q+1).trim.split(",").head
      RowFactory.create(category, text)
    }
    val schema = StructType(Array(StructField("category", StringType, false), StructField("text", StringType, false)))
    import sparkSession.implicits._
    sparkSession.createDataFrame(rows, schema).as[Document]
  }

  def readHSD(sparkSession: SparkSession, path: String): Dataset[Document] = {
    import sparkSession.implicits._
    sparkSession.read.json(path).select("category", "withoutAccent")
      .withColumnRenamed("withoutAccent", "text").as[Document]
  }

  /**
    * Reads 5cats dataset containing a collectin of vnExpress articles.
    * This function is used in preparation for the FAIR'20 paper.
    *
    * @param sparkSession
    * @param path a directory contain JSON files (see /opt/data/vne/5cats.utf8/)
    * @param percentage
    * @return a dataset of documents
    */
  def read5Cats(sparkSession: SparkSession, path: String, percentage: Double = 1.0): Dataset[Document] = {
        // each .json file is read to a df and these dfs are concatenated to form a big df
    val filenames = new File(path).list().filter(_.endsWith(".json"))
    val dfs = filenames.map(f => sparkSession.read.json(path + f))
    val input = dfs.reduce(_ union _)
    val textSet = input.sample(percentage)

    import sparkSession.implicits._
    val categories = textSet.select("category").map(row => row.getString(0)).distinct.collect().sorted
    val labels = categories.zipWithIndex.toMap
    val numLabels = labels.size
    println(s"Found ${numLabels} classes")
    textSet.as[Document]
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
      opt[String]('c', "classifier").action((x, conf) => conf.copy(classifier = x)).text("classifier, either mlr/mlp/rfc")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[String]('h', "hiddenUnits").action((x, conf) => conf.copy(hiddenUnits = x)).text("hidden units in MLP")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[Int]('t', "numTrees").action((x, conf) => conf.copy(numTrees = x)).text("number of trees if using RFC, default is 256")
      opt[Int]('e', "maxDepth").action((x, conf) => conf.copy(maxDepth = x)).text("max tree depth if using RFC, default is 15")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("data path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/tcl/'")
      opt[String]('i', "input").action((x, conf) => conf.copy(input = x)).text("input path")
      opt[String]('o', "output").action((x, conf) => conf.copy(output = x)).text("output path")
    }
    parser.parse(args, ConfigTCL()) match {
      case Some(config) =>
        val sparkSession = SparkSession.builder().appName(getClass.getName).master(config.master).getOrCreate()
        implicit val formats = Serialization.formats(NoTypeHints)
        logger.info(Serialization.writePretty(config))
        val tcl = new Classifier(sparkSession.sparkContext, config)
        config.mode match {
          case "train" => {
            val dataset = tcl.createDataset(config.dataPath)
            val Array(training, test) = dataset.randomSplit(Array(0.8, 0.2), seed = 20150909)
            training.show()
            tcl.train(training)
            tcl.eval(training)
            test.show()
            tcl.eval(test)
          }
          case "eval" => 
          case "sample" => sampling(sparkSession, "dat/vne/5cats.txt", "dat/vne/5catsSample", 0.01)
          case "predict" => tcl.predict(config.input, config.output)
          case "trainShinra" => 
            val numberOfSentences = 5
            val trainingDataset = readSHINRA(sparkSession, config.dataPath, numberOfSentences)
            trainingDataset.show()
            val model = tcl.train(trainingDataset)
            tcl.eval(model, trainingDataset)
          case "evalShinra" =>
            val numberOfSentences = 3
            val model = PipelineModel.load(config.modelPath + "/" + config.classifier.toLowerCase())
            val trainingDataset = readSHINRA(sparkSession, config.dataPath, numberOfSentences)
            val devDataset = readSHINRA(sparkSession, "/opt/data/shinra/dev.txt", numberOfSentences)
            val testDataset = readSHINRA(sparkSession, "/opt/data/shinra/test.txt", numberOfSentences)
            tcl.eval(model, trainingDataset)
            tcl.eval(model, devDataset)
            tcl.eval(model, testDataset)
          case "trainHSD" => 
            val shd = readHSD(sparkSession, "dat/hsd/withoutAccent")
            val Array(a, b) = shd.randomSplit(Array(0.8, 0.2), seed = 20150909)
            a.show()
            tcl.train(a)
            tcl.eval(a)
            b.show()
            tcl.eval(b)
          case "evalHSD" => 
            val shd = readHSD(sparkSession, "dat/hsd/withoutAccent")
            val stats = shd.groupBy("category").count()
            stats.show(false)
            val Array(a, b) = shd.randomSplit(Array(0.8, 0.2), seed = 20150909)
            a.show()
            tcl.eval(a)
            b.show()
            tcl.eval(b)
          case "5cats" => 
            val dataset = read5Cats(sparkSession, "/opt/data/vne/5cats.utf8/", 0.1)
            val Array(training, test) = dataset.randomSplit(Array(0.8, 0.2), seed = 20150909)
            training.show()
            tcl.train(training)
            tcl.eval(training)
            test.show()
            tcl.eval(test)
        }
        sparkSession.stop()
      case None =>
      }
  }
}
