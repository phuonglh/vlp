package vlp.woz.jsl

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach
import com.johnsnowlabs.nlp.embeddings.{BertEmbeddings, DeBertaEmbeddings, DistilBertEmbeddings, UniversalSentenceEncoder, XlmRoBertaEmbeddings}
import org.apache.spark.ml.{Pipeline, PipelineModel}
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
  */

object MultilabelClassifier {
  implicit val formats = Serialization.formats(NoTypeHints)

  def train(config: ConfigJSL, df: DataFrame): PipelineModel = {
    val document = new DocumentAssembler().setInputCol("utterance").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
    val embeddings = config.modelType match {
      case "b" => BertEmbeddings.pretrained("bert_embeddings_bert_base_vi_cased", "vi").setInputCols(Array("token", "document")).setOutputCol("embeddings")
      case "d" => DeBertaEmbeddings.pretrained("deberta_embeddings_spm_vie", "vie").setInputCols(Array("token", "document")).setOutputCol("embeddings")
      case "l" => DistilBertEmbeddings.pretrained("distilbert_embeddings_base_multilingual_cased", "xx").setInputCols(Array("token", "document")).setOutputCol("embeddings")
      case "u" => UniversalSentenceEncoder.pretrained().setInputCols("document").setOutputCol("embeddings")
      case "x" => XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_afriberta_base", "xx").setInputCols(Array("token", "document")).setOutputCol("embeddings")
      case _ => UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx").setInputCols("document").setOutputCol("embeddings")
    }
    val classifier = new MultiClassifierDLApproach().setInputCols("embeddings").setOutputCol("category").setLabelColumn("actNames")
      .setBatchSize(config.batchSize).setMaxEpochs(config.epochs).setLr(config.learningRate.toFloat)
      .setThreshold(0.5f)
      .setValidationSplit(0.1f)
    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings, classifier)) 
    val model = pipeline.fit(df)
    return model
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[ConfigJSL](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 8g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
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

        val df = spark.read.json(config.trainPath)
        df.show()
        df.printSchema()
        config.mode match {
          case "train" =>
            val model = train(config, df)
            val output = model.transform(df)
            output.printSchema
            output.show()
            model.write.overwrite.save(config.modelPath)
          case "eval" => 
        }

        sc.stop()
      case None => {}
    }

  }
}