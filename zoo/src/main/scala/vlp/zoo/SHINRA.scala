package vlp.zoo

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}
import org.slf4j.LoggerFactory

import com.intel.analytics.bigdl.optim.{Adam, Top1Accuracy, Top5Accuracy}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.feature.text.{TextFeature, TextSet}
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import scopt.OptionParser
import com.intel.analytics.zoo.pipeline.api.keras.metrics.SparseCategoricalAccuracy
import java.text.DateFormat
import java.text.SimpleDateFormat
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.sql.RowFactory

import vlp.tok.TokenizerTransformer
import com.intel.analytics.zoo.pipeline.api.keras.metrics.Accuracy

import org.json4s._
import org.json4s.jackson.Serialization._
import org.json4s.jackson.Serialization
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import scala.io.Source
import java.nio.charset.StandardCharsets
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.Sample
import org.apache.spark.sql.SaveMode
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.Pipeline
import com.intel.analytics.zoo.common.NNContext

case class ConfigSHINRA(
  master: String = "local[*]",
  mode: String = "eval",
  executorMemory: String = "8g",
  language: String = "en",
  dataPath: String = "/opt/data/shinra/",
  embeddingPath: String = "/opt/data/emb/",
  modelPath: String = "dat/zoo/tcl/shi/",
  numFeatures: Int = 100000,
  encoder: String = "cnn",
  embeddingDimension: Int = 100,
  encoderOutputDimension: Int = 256,
  maxSequenceLength: Int = 256,
  batchSize: Int = 64,
  epochs: Int = 50,
  learningRate: Double = 0.001,
  percentage: Double = 1.0,
  partitions: Int = 20,
  minFrequency: Int = 2,
  verbose: Boolean = false,
  inputCol: String = "body",
  classCol: String = "clazz"
)

case class ENE(ENE_id: String, ENE_name: String, score: Double = 1.0)
case class Result(
  pageid: String,
  title: String,
  lang: String,
  ENEs: List[ENE]
)

class LanguagePack(config: ConfigSHINRA) {
  def modelPath = Paths.get(config.modelPath, config.language, config.encoder).toString
  def dataPath = Paths.get(config.dataPath, config.language).toString()
  def embeddingPath = Paths.get(config.embeddingPath, config.language, "glove.6B." + config.embeddingDimension.toString() + "d.txt").toString 
}

/**
  * Deep learning based text classifier. We can use CNN, LSTM or GRU architecture.
  *
  * @param sparkContext
  * @param config
  */
class SHINRA(val sparkContext: SparkContext, val config: ConfigSHINRA) {
  final val logger = LoggerFactory.getLogger(getClass.getName)
  val sparkSession = SparkSession.builder().getOrCreate()
  import sparkSession.implicits._
  implicit val formats = Serialization.formats(NoTypeHints)
  final val languagePack = new LanguagePack(config)
  
  /**
    * Trains a neural text classifier on a text set and validate on a validation set. Trained model and word index are saved to
    * an external path specified in the configuration.
    * @param trainingSet
    * @param validationSet
    */
  def train(trainingSet: TextSet, validationSet: TextSet): TextClassifier[Float] = {
    logger.info("Processing text data set...")
    val transformedTrainingSet = trainingSet.tokenize()
      .word2idx(10, maxWordsNum = config.numFeatures, minFreq = config.minFrequency)
      .shapeSequence(config.maxSequenceLength)
      .generateSample()
    val wordIndex = transformedTrainingSet.getWordIndex
    transformedTrainingSet.saveWordIndex(languagePack.modelPath + ".dict.txt")
    logger.info("Word index created.")

    val classifier = TextClassifier(SHINRA.numLabels, languagePack.embeddingPath, wordIndex, config.maxSequenceLength, config.encoder, config.encoderOutputDimension)
    val date = new SimpleDateFormat("yyyy-MM-dd.HHmmss").format(new java.util.Date())
    classifier.setTensorBoard(Paths.get("/tmp/zoo/tcl/shi", config.language, "/").toString, appName = config.encoder + "/" + date)
    classifier.compile(
      optimizer = new Adam(learningRate = config.learningRate),
      loss = SparseCategoricalCrossEntropy[Float](),
      metrics = List(new SparseCategoricalAccuracy[Float]())
    )
    if (config.percentage < 1) {
      logger.info("Preparing validation set...")
      val transformedValidationSet = validationSet.tokenize()
        .setWordIndex(wordIndex).word2idx()
        .shapeSequence(config.maxSequenceLength)
        .generateSample()
      classifier.fit(transformedTrainingSet, batchSize = config.batchSize, nbEpoch = config.epochs, transformedValidationSet)
    } else classifier.fit(transformedTrainingSet, batchSize = config.batchSize, nbEpoch = config.epochs, transformedTrainingSet)

    classifier.saveModel(languagePack.modelPath, overWrite = true)
    logger.info("Finish training model and saving word dictionary.")
    classifier
  }

  def predict(textSet: TextSet, classifier: TextClassifier[Float], docIds: Array[String] = Array.empty[String], outputFile: String = ""): TextSet = {
    val transformedTextSet = textSet.tokenize().loadWordIndex(languagePack.modelPath + ".dict.txt").word2idx()
      .shapeSequence(config.maxSequenceLength).generateSample()
      DataSet
      TextFeature
    val result = classifier.predict(transformedTextSet, batchPerThread = config.partitions)
    val predictedClasses = classifier.predictClasses(transformedTextSet.toDistributed().rdd.map(_.getSample))
    if (outputFile.nonEmpty) {
      // restore the label map from an external JSON file
      val labelPath = languagePack.modelPath + ".labels.json"
      implicit val formats = Serialization.formats(NoTypeHints)
      import scala.collection.JavaConversions._
      val mapSt = Files.readAllLines(Paths.get(labelPath))(0)
      val objectMapper = new ObjectMapper() with ScalaObjectMapper
      objectMapper.registerModule(DefaultScalaModule)
      val labels = objectMapper.readValue(mapSt, classOf[Map[String, Int]])
      val clazzMap = labels.keySet.map(key => (labels(key), key)).toMap[Int, String]
      val prediction = predictedClasses.collect().map(k => clazzMap(k))
      val output = docIds.zip(prediction).map(pair => Result(pair._1, "", "", List(ENE(pair._2, "", 0.0))))
            .map(result => Serialization.write(result))
      import scala.collection.JavaConversions._  
      Files.write(Paths.get(outputFile), output.toList, StandardCharsets.UTF_8)
    }
    result
  }
}

object SHINRA {
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  var numLabels = 0

  /**
    * Reads a corpus in JSONL format, extract 'text' and 'category' fields to build a text set.
    *
    * @param sparkSession
    * @param config
    * @return a data frame of two columns (category, text)
    */
  def readJsonData(sparkSession: SparkSession, config: ConfigSHINRA): TextSet = {
    val languagePack = new LanguagePack(config)
    val input = sparkSession.read.json(languagePack.dataPath)
    val textSet = if (config.percentage < 1.0) input.sample(config.percentage) else input

    import sparkSession.implicits._
    val categories = textSet.select(config.classCol).flatMap(row => row.getString(0).split(",")).distinct.collect().sorted
    val labelPath = languagePack.modelPath + ".labels.json"
    implicit val formats = Serialization.formats(NoTypeHints)
    val labels = if (config.mode == "train") {
      val map = categories.zipWithIndex.toMap
      numLabels = map.size
      println(s"Found ${numLabels} classes")
      println(map.mkString(", "))
      // write label map to an external JSON file
      val mapSt = Serialization.write(map)
      Files.write(Paths.get(labelPath), mapSt.getBytes(), StandardOpenOption.CREATE)
      map
    } else {
      // restore the label map from an external JSON file
      import scala.collection.JavaConversions._
      val mapSt = Files.readAllLines(Paths.get(labelPath))(0)
      val objectMapper = new ObjectMapper() with ScalaObjectMapper
      objectMapper.registerModule(DefaultScalaModule)
      objectMapper.readValue(mapSt, classOf[Map[String, Int]])
    }
    println("Creating text set. Please wait...")
    val textRDD = textSet.select(config.classCol, config.inputCol).rdd.map(row => {
      val content = row.getString(1)
      val label = row.getString(0).split(",").head
      TextFeature(content, labels(label))
    })
    TextSet.rdd(textRDD)
  }

  def text2Json(sparkSession: SparkSession, config: ConfigSHINRA): Unit = {
    val inputPath = Paths.get(config.dataPath, config.language + ".txt").toString()
    val outputPath = Paths.get(config.dataPath, config.language).toString()
    import sparkSession.implicits._
    val df = sparkSession.read.text(inputPath)
    val rdd = df.rdd.map(row => {
      val parts = row.getString(0).split("\t")
      if (parts.size == 4)
        RowFactory.create(parts(2), parts(3))
      else {
        println(row.getString(0) + "\n")
        RowFactory.create(parts(2), "")
      }
    }).filter(row => row.getString(1).nonEmpty)

    val schema = StructType(Array(StructField("clazz", StringType, false), StructField("text", StringType, false)))
    val input = sparkSession.createDataFrame(rdd, schema)
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("""[\s+.,·:\)\(\]\[?;~"`'»«’↑\u200e\u200b\ufeff\\]+""")
    val lang = config.language match {
      case "en" => "english"
      case "fr" => "french"
      case _ => "english"
    }
    val remover = new StopWordsRemover().setInputCol("tokens").setOutputCol("words").setStopWords(StopWordsRemover.loadDefaultStopWords(lang))
    val temp = remover.transform(tokenizer.transform(input)).select("clazz", "words")
    import org.apache.spark.sql.functions.concat_ws
    val output = temp.withColumn("body", concat_ws(" ", $"words")).select("clazz", "body")
    output.repartition(config.partitions).write.mode(SaveMode.Overwrite).json(outputPath)
  }

  def statistics(sparkSession: SparkSession, config: ConfigSHINRA): Unit = {
    val languagePack = new LanguagePack(config)
    val textSet = sparkSession.read.json(languagePack.dataPath)
    import sparkSession.implicits._
    val categories = textSet.groupBy(config.classCol).count().sort($"count".desc)
    categories.show(100)
    println("#(records) = " + categories.count())
  }

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[ConfigSHINRA]("zoo.tcl.SHINRA") {
      head("vlp.zoo.SHINRA", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/predict")
      opt[String]('e', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("data path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/zoo/tcl/'")
      opt[Int]('w', "embeddingDimension").action((x, conf) => conf.copy(embeddingDimension = x)).text("embedding dimension 50/100/200/300")
      opt[String]('t', "encoder").action((x, conf) => conf.copy(encoder = x)).text("type of encoder, either cnn, lstm or gru")
      opt[Int]('o', "encoderOutputDimension").action((x, conf) => conf.copy(encoderOutputDimension = x)).text("output dimension of the encoder")
      opt[Int]('l', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("maximum sequence length for a text")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('r', "percentage").action((x, conf) => conf.copy(percentage = x)).text("ratio of the training set to use, default is 1.0")
      opt[String]('x', "inputCol").action((x, conf) => conf.copy(inputCol = x)).text("input column")
      opt[String]('y', "classCol").action((x, conf) => conf.copy(classCol = x)).text("class column")

    }
    parser.parse(args, ConfigSHINRA()) match {
      case Some(config) =>
      val sparkConfig = Engine.createSparkConf()
        .setMaster(config.master)
        .set("spark.executor.memory", config.executorMemory)
        .setAppName("SHINRA")
      val sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate()
      val sparkContext = sparkSession.sparkContext
      NNContext.initNNContext(sparkConfig)

      val app = new SHINRA(sparkSession.sparkContext, config)
      val languagePack = new LanguagePack(config)
      val validationMethods = Array(new Accuracy[Float]())

      config.mode match {
        case "json" =>
          text2Json(sparkSession, config)
        case "train" =>
          val textSet = readJsonData(sparkSession, config).toDistributed(sparkContext, config.partitions)
          if (config.percentage < 1.0) {
            val Array(trainingSet, validationSet) = textSet.randomSplit(Array(config.percentage, 1 - config.percentage))
            app.train(trainingSet, validationSet)
          } else app.train(textSet, textSet)
        case "eval" =>
          val classifier = TextClassifier.loadModel[Float](languagePack.modelPath)
          classifier.setEvaluateStatus()
          val textSet = readJsonData(sparkSession, config).toDistributed(sparkContext, config.partitions)
          val prediction = app.predict(textSet, classifier)
          var accuracy = classifier.evaluate(prediction.toDistributed().rdd.map(_.getSample), validationMethods)
          println("validation accuracy = " + accuracy.mkString(", "))
        case "predict" =>
          import sparkSession.implicits._
          val df = sparkSession.read.text("dat/shi/" + config.language + ".txt").rdd.filter(row => row.getString(0).trim.nonEmpty).map(row => {
            val parts = (row.getString(0) + "\tNA").split("\t")
            RowFactory.create(parts: _*)
          })
          val schema = StructType(Array(StructField("pageid", StringType, false), StructField("text", StringType, false)))
          val input = sparkSession.createDataFrame(df, schema)
          val docIds = input.select("pageid").map(row => row.getString(0)).collect()
          val xs = if (config.language == "vi") {
            val tokenizer = new TokenizerTransformer().setInputCol("text").setOutputCol("body").setConvertNumber(true).setToLowercase(true)
            tokenizer.transform(input)
          } else {
            val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern("""[\s+.,·:\)\(\]\[?;~"`'»«’↑\u200e\u200b\ufeff\\]+""")
            val lang = config.language match {
              case "en" => "english"
              case "fr" => "french"
              case _ => "english"
            }
            val remover = new StopWordsRemover().setInputCol("tokens").setOutputCol("words").setStopWords(StopWordsRemover.loadDefaultStopWords(lang))
            val temp = remover.transform(tokenizer.transform(input))
            import org.apache.spark.sql.functions.concat_ws
            temp.withColumn("body", concat_ws(" ", $"words"))
          }
          xs.show()
          val textRDD = xs.select("body").rdd.map(row => {
            val content = row.getString(0).split("\\s+").toArray
            TextFeature(content.mkString(" "), -1)
          })
          val classifier = TextClassifier.loadModel[Float](languagePack.modelPath)
          classifier.setEvaluateStatus()
          val textSet = TextSet.rdd(textRDD)
          app.predict(textSet, classifier, docIds, "dat/shi/" + config.language + ".json." + config.encoder)
        case "stat" => statistics(sparkSession, config)
      }
      sparkSession.stop()
      case None =>
    }
  }
}