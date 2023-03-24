package vlp.woz.jsl

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.functions._
import com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach
import com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings
import com.johnsnowlabs.nlp.embeddings.{DeBertaEmbeddings, DistilBertEmbeddings, XlnetEmbeddings}
import com.johnsnowlabs.nlp.embeddings.{BertSentenceEmbeddings, RoBertaSentenceEmbeddings, XlmRoBertaSentenceEmbeddings, UniversalSentenceEncoder}
import com.johnsnowlabs.nlp.Annotation
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import vlp.woz.DialogReader

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

import java.nio.file.{Files, Paths, StandardOpenOption}

/**
  * phuonglh, March 2023
  * 
  * Experiments with multiple pretrained models for sentence embeddings and their 
  * effectiveness in dialogue act classification.
  * 
  */

object MultilabelClassifier {
  implicit val formats = Serialization.formats(NoTypeHints)

  def train(config: ConfigJSL, df: DataFrame): PipelineModel = {
    // use a vectorizer to get label vocab
    val actVectorizer = new CountVectorizer().setInputCol("actNames").setOutputCol("ys")
    val document = new DocumentAssembler().setInputCol("utterance").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
    val embeddings = config.modelType match {
      case "b" => BertSentenceEmbeddings.pretrained("sent_bert_multi_cased", "xx").setInputCols("document").setOutputCol("embeddings")
      case "u" => UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx").setInputCols("document").setOutputCol("embeddings")
      case "r" => RoBertaSentenceEmbeddings.pretrained().setInputCols("document").setOutputCol("embeddings") // this is for English only
      case "x" => XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_base", "xx").setInputCols("document").setOutputCol("embeddings")
      case "d" => if (config.language == "vi")
          DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("xs")
        else 
          DeBertaEmbeddings.pretrained("deberta_v3_base", "en").setInputCols("document", "token").setOutputCol("xs") 
      case "s" => if (config.language == "vi")
          DistilBertEmbeddings.pretrained("distilbert_base_cased", "vi").setInputCols("document", "token").setOutputCol("xs")
        else 
          DistilBertEmbeddings.pretrained("distilbert_base_cased", "en").setInputCols("document", "token").setOutputCol("xs") 
      case _ => UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx").setInputCols("document").setOutputCol("embeddings")
    }
    var stages = Array(actVectorizer, document, tokenizer, embeddings)
    val sentenceEmbedding = new SentenceEmbeddings().setInputCols("document", "xs").setOutputCol("embeddings")
    if (Set("d", "s").contains(config.modelType)) 
      stages = stages ++ Array(sentenceEmbedding)
    val classifier = new MultiClassifierDLApproach().setInputCols("embeddings").setOutputCol("category").setLabelColumn("actNames")
      .setBatchSize(config.batchSize).setMaxEpochs(config.epochs).setLr(config.learningRate.toFloat)
      .setThreshold(0.4f)
      .setValidationSplit(0.1f)
    stages = stages ++ Array(classifier)
    val pipeline = new Pipeline().setStages(stages)
    val model = pipeline.fit(df)
    return model
  }

  def predict(df: DataFrame, model: PipelineModel, config: ConfigJSL, split: String): DataFrame = {
    val ef = model.transform(df)
    val ff = ef.mapAnnotationsCol("category", "prediction", "category", (a: Seq[Annotation]) => if (a.nonEmpty) a.map(_.result) else List.empty[String])
    ff.select("actNames", "prediction")
  }

  def evaluate(result: DataFrame, labelIndex: Map[String, Double], config: ConfigJSL, split: String): Score = {
    // predict
    val predictionsAndLabels = result.rdd.map { case row => 
      (row.getAs[Seq[String]](1).toArray, row.getAs[Seq[String]](0).toArray)
    }
    // convert to Double value
    val zy = predictionsAndLabels.map { case (zs, ys) =>
      (zs.map(labelIndex(_)), ys.map(labelIndex(_)))
    }
    val metrics = new MultilabelMetrics(zy)
    val labelSize = 0 // TODO
    val precisionByLabel = Array.fill(labelSize)(0d)
    val recallByLabel = Array.fill(labelSize)(0d)
    val fMeasureByLabel = Array.fill(labelSize)(0d)
    val ls = metrics.labels
    println(ls.mkString(", "))
    ls.foreach { k => 
      precisionByLabel(k.toInt) = metrics.precision(k)
      recallByLabel(k.toInt) = metrics.recall(k)
      fMeasureByLabel(k.toInt) = metrics.f1Measure(k)
    }
    Score(
      config.modelType, split,
      metrics.accuracy, metrics.f1Measure, metrics.microF1Measure, metrics.microPrecision, metrics.microRecall,
      precisionByLabel, recallByLabel, fMeasureByLabel
    )
  }

  def saveScore(score: Score, path: String) = {
    var content = Serialization.writePretty(score) + ",\n"
    Files.write(Paths.get(path), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[ConfigJSL](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 16g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language {en, vi}")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 5E-4")
      opt[String]('d', "trainPath").action((x, conf) => conf.copy(trainPath = x)).text("training data directory")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[String]('s', "scorePath").action((x, conf) => conf.copy(scorePath = x)).text("score path")
    }
    opts.parse(args, ConfigJSL()) match {
      case Some(config) =>
        val conf = new SparkConf().setAppName(getClass().getName()).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        // sc.setLogLevel("ERROR")

        val df = spark.read.json(config.trainPath)
        df.show()
        df.printSchema()
        config.mode match {
          case "train" =>
            val model = train(config, df)
            val output = model.transform(df)
            output.printSchema
            output.show()
            model.write.overwrite.save(config.modelPath + "/" + config.modelType)
          case "predict" =>
            val model = PipelineModel.load(config.modelPath + "/" + config.modelType)
            val ef = predict(df, model, config, "all")
            ef.show(false)
          case "eval" => 
            val model = PipelineModel.load(config.modelPath + "/" + config.modelType)
            val labels = model.stages(0).asInstanceOf[CountVectorizerModel].vocabulary
            val labelIndex = labels.zipWithIndex.toMap.mapValues(_.toDouble)
            val ef = predict(df, model, config, "all")
            ef.show(false)
            val score = evaluate(ef, labelIndex, config, "all")
            println(Serialization.writePretty(score))
        }

        sc.stop()
      case None => {}
    }

  }
}