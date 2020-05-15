package vlp.vdr

import org.apache.log4j._
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{HashingTF, RegexTokenizer, StringIndexer, StringIndexerModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory
import scopt.OptionParser

/**
  * phuonglh, 10/31/17, 18:59
  */
class Restorer(sparkSession: SparkSession, config: ConfigVDR) {
  final val logger = LoggerFactory.getLogger(getClass.getName)

  def train(input: DataFrame): PipelineModel = {
    input.cache()
    logger.info("Training the model. Please wait...")
    val x = new Sequencer().setInputCol("sentence").setOutputCol("x").setDiacritic(false)
    val y = new Sequencer().setInputCol("sentence").setOutputCol("y")
    val sampler = new Sampler().setFeatureCol("f").setLabelCol("c").setMarkovOrder(config.markovOrder)
    val labelIndexer = new StringIndexer().setInputCol("c").setOutputCol("label").setHandleInvalid("skip")
    val tokenizer = new RegexTokenizer().setInputCol("f").setOutputCol("tokens").setToLowercase(false)
    val hashingTF = new HashingTF().setInputCol("tokens").setOutputCol("features").setNumFeatures(config.numFeatures)
    val classifier = new LogisticRegression().setMaxIter(config.iterations).setStandardization(false).setRegParam(config.lambda).setTol(1E-5)
    
    val pipeline = new Pipeline().setStages(Array(x, y, sampler, labelIndexer, tokenizer, hashingTF, classifier))
    val model = pipeline.fit(input)
    model.write.overwrite().save(config.modelPath)
    if (config.verbose) {
      logger.info("#(training sentences) = " + input.count())
      logger.info("#(labels) = " + model.stages(3).asInstanceOf[StringIndexerModel].labels.size)
      val labels = model.stages(3).asInstanceOf[StringIndexerModel].labels
      logger.info(labels.mkString(", "))
    }
    model
  }

  /**
    * Accented letter based evaluation.
    * @param input
    */
  def evaluate(input: DataFrame): Unit = {
    val model = PipelineModel.load(config.modelPath)
    val output = model.transform(input)
    output.show(10)
    logger.info("#(samples) = " + output.count())
    import sparkSession.implicits._
    val predictionAndLabels = output.select("label", "prediction").map(row => (row.getDouble(0), row.getDouble(1))).rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val scores = (metrics.accuracy, metrics.weightedFMeasure)
    logger.info(s" (A, F) scores = $scores")

    if (config.verbose) {
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

  /**
    * Accented token-based evaluation  
    * @param sentences
    */
  def evaluate(sentences: Seq[String], mappingPath: String): Unit = {
    import sparkSession.implicits._
    val input = sentences.toDF("sentence")
    val remover = new Remover().setInputCol("sentence").setOutputCol("x")
    val predictions = tag(PipelineModel.load(config.modelPath), remover.transform(input), config.greedy, mappingPath)
    predictions.show(10, false)
    val x = new RegexTokenizer().setToLowercase(false).setInputCol("sentence").setOutputCol("xs")
    val b = new BestSolution().setInputCol("y").setOutputCol("b")
    val y = new RegexTokenizer().setToLowercase(false).setInputCol("b").setOutputCol("ys")
    val pipeline = new Pipeline()
    pipeline.setStages(Array(x, b, y))
    val last = pipeline.fit(predictions).transform(predictions)
    last.select("xs", "ys").show(10, false)
    val tokens = last.select("xs", "ys")
      .map(row => (row.getAs[Seq[String]](0), row.getAs[Seq[String]](1)))
      .flatMap(yz => yz._1.zip(yz._2)).collect()
      
    val numTokens = tokens.size
    val numCorrectTokens = tokens.map(p => if (p._1 == p._2) 1 else 0).sum
    logger.info("Token accuracy = " + numCorrectTokens.toDouble / numTokens)
    if (config.verbose) {
      tokens.filter(p => p._1 != p._2).map(p => logger.info(p.toString()))
    }
  }
  
  def tag(model: PipelineModel, input: DataFrame, greedyDecoding: Boolean, mappingsPath: String): DataFrame = {
    val labels = model.stages(3).asInstanceOf[StringIndexerModel].labels
    val featureTypes = model.stages(2).asInstanceOf[Sampler].getFeatureTypes
    val markovOrder = model.stages(2).asInstanceOf[Sampler].getMarkovOrder
    val numFeatures = model.stages(5).asInstanceOf[HashingTF].getNumFeatures
    val (weights, intercepts) = {
      val mlr = model.stages(6).asInstanceOf[LogisticRegressionModel]
      (mlr.coefficientMatrix, mlr.interceptVector)
    }
    val decoder = new Decoder(labels, featureTypes, markovOrder, numFeatures, weights, intercepts)
      .setInputCol("x").setOutputCol("y").setGreedy(greedyDecoding)
    val output = decoder.transform(input)
    if (mappingsPath.isEmpty)
      output
    else {
      val mapper = new Mapper(mappingsPath).setInputCol("x").setPredictionCol("y").setOutputCol("z")
      val mappedOutput = mapper.transform(output)
      mappedOutput
    }
  }
  
  def tag(model: PipelineModel, sentences: Seq[String], greedyDecoding: Boolean, mappingPath: String): Seq[Seq[String]] = {
    import sparkSession.implicits._
    val output = tag(model, sentences.toDF("x"), greedyDecoding, mappingPath)
    if (mappingPath.isEmpty)
      output.select("y").map(row => row.getSeq[String](0)).collect()
    else
      output.select("z").map(row => row.getSeq[String](0)).collect()
  }
  
}

object Restorer {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val parser = new OptionParser[ConfigVDR]("vlp.vdr") {
      head("vlp.vdr", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[Unit]('g', "greedy").action((_, conf) => conf.copy(greedy = true)).text("greedy inference")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("training data path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/vdr/'")
      opt[String]('i', "input").action((x, conf) => conf.copy(input = x)).text("input path")
    }

    parser.parse(args, ConfigVDR()) match {
      case Some(config) =>
        val sparkSession = SparkSession.builder().appName(getClass.getName).master(config.master)
          .config("spark.kryoserializer.buffer.max.mb", "512").getOrCreate()
        val restorer = new Restorer(sparkSession, config)
        import sparkSession.implicits._
        val input = IO.readSentences(config.dataPath).toDF("sentence")
        val Array(trainingData, testData) = input.randomSplit(Array(0.8, 0.2), 150909)
        config.mode match {
          case "train" =>
            restorer.train(trainingData)
            restorer.evaluate(testData)
            restorer.evaluate(trainingData)
          case "eval" =>
            restorer.evaluate(testData)
            restorer.evaluate(trainingData)
          case "tag" =>
            val sentences = Seq("quan ao thoi trang", "danh lam thang canh", "sinh vien khong hoc bai tot")
            val model = PipelineModel.load(config.modelPath)
            import sparkSession.implicits._
            val output = restorer.tag(model, sentences.toDF("x"), config.greedy, config.mappingResourcePath)
            output.show(false)
            val output2 = restorer.tag(model, sentences, config.greedy, config.mappingResourcePath)
            output2.foreach(println)
          case _ =>
        }
        sparkSession.stop()
      case None =>
    }
  }
}
