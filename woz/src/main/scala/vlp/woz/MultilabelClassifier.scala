package vlp.woz

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

  def train(config: ConfigJSL, trainingDF: DataFrame, developmentDF: DataFrame): PipelineModel = {
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
      .setThreshold(config.threshold)
    // train a preprocessor 
    val preprocessor = new Pipeline().setStages(stages)
    val preprocessorModel = preprocessor.fit(trainingDF)
    // use the preprocessor pipeline to transform the development set 
    val df = preprocessorModel.transform(developmentDF)
    df.write.mode("overwrite").parquet(config.validPath)
    classifier.setTestDataset(config.validPath)
    // train the whole pipeline and return a model
    stages = stages ++ Array(classifier)
    val pipeline = new Pipeline().setStages(stages)
    val model = pipeline.fit(trainingDF)
    return model    
  }

  def predict(df: DataFrame, model: PipelineModel, config: ConfigJSL, split: String): DataFrame = {
    val ef = model.transform(df)
    val ff = ef.mapAnnotationsCol("category", "prediction", "category", (a: Seq[Annotation]) => if (a.nonEmpty) a.map(_.result) else List.empty[String])
    ff.select("prediction", "actNames")
  }

  def evaluate(result: DataFrame, config: ConfigJSL, split: String): Score = {
    val predictionsAndLabels = result.rdd.map { case row => 
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
    Score(
      config.language,
      config.modelType, split,
      metrics.accuracy, metrics.f1Measure, 
      metrics.microF1Measure, metrics.microPrecision, metrics.microRecall,
      precisionByLabel, recallByLabel, fMeasureByLabel
    )
  }

  def saveScore(score: Score, path: String) = {
    var content = Serialization.writePretty(score) + ",\n"
    Files.write(Paths.get(path), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
  }

  def evaluate(df: DataFrame, model: PipelineModel, config: ConfigJSL, split: String): Unit = {
    val ef = predict(df, model, config, split)
    val labels = model.stages(0).asInstanceOf[CountVectorizerModel].vocabulary
    val labelIndex = labels.zipWithIndex.toMap
    val seq1 = new Sequencer(labelIndex).setInputCol("prediction").setOutputCol("zs")
    val seq2 = new Sequencer(labelIndex).setInputCol("actNames").setOutputCol("ys")
    val ff = seq2.transform(seq1.transform(ef))
    ff.show()
    val result = ff.select("zs", "ys")
    val score = evaluate(result, config, split)
    println(Serialization.writePretty(score))
    saveScore(score, config.scorePath)
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
        sc.setLogLevel("ERROR")
        val Array(trainingDF, developmentDF, testDF) = if (config.language == "vi") {
          val df = spark.read.json(config.trainPath)
          df.randomSplit(Array(0.8, 0.1, 0.1))
        } else {
          Array(spark.read.json(config.trainPath), spark.read.json(config.devPath), spark.read.json(config.testPath))
        }
        testDF.show()
        testDF.printSchema()
        val modelPath = config.modelPath + "/" + config.language + "/" + config.modelType
        config.mode match {
          case "train" =>
            val model = train(config, trainingDF, developmentDF)
            val output = model.transform(developmentDF)
            output.printSchema
            output.show()
            model.write.overwrite.save(modelPath)
            evaluate(trainingDF, model, config, "train")
            evaluate(developmentDF, model, config, "valid")
            evaluate(testDF, model, config, "test")
          case "predict" =>
            val model = PipelineModel.load(modelPath)
            val ef = predict(developmentDF, model, config, "valid")
            ef.show(false)
          case "eval" => 
            val model = PipelineModel.load(modelPath)
            evaluate(trainingDF, model, config, "train")
            evaluate(developmentDF, model, config, "valid")
            evaluate(testDF, model, config, "test")
        }

        sc.stop()
      case None => {}
    }

  }
}