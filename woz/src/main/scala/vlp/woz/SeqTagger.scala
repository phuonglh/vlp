package vlp.woz

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.functions._
import com.johnsnowlabs.nlp.Annotation
import org.apache.spark.ml.{Pipeline, PipelineModel}
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
  * Experiments with multiple pretrained models for token embeddings and their 
  * effectiveness in sequence tagging.
  * 
  */

object SeqTagger {
  implicit val formats = Serialization.formats(NoTypeHints)

  def train(config: ConfigJSL, trainingDF: DataFrame, developmentDF: DataFrame): PipelineModel = {
    // use a vectorizer to get label vocab
    val document = new DocumentAssembler().setInputCol("utterance").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
    val classifier = RoBertaForTokenClassification.pretrained().setInputCols("document", "token").setOutputCol("named_entity").setCaseSensitive(true) 
    val nerConverter = new NerConverter().setInputCols("document", "token", "named_entity").setOutputCol("ner_converter")
    val finisher = new Finisher().setInputCols("named_entity", "ner_converter").setCleanAnnotations(false)
        
    val pipeline = new Pipeline().setStages(Array(document, tokenizer, classifier, nerConverter, finisher))
    val model = pipeline.fit(trainingDF)
    return model    
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
          Array(spark.read.json(config.devPath), spark.read.json(config.devPath), spark.read.json(config.testPath))
        }
        testDF.show()
        testDF.printSchema()
        val modelPath = "bin/seq/" + config.language + "/" + config.modelType
        config.mode match {
          case "train" =>
            val model = train(config, trainingDF, developmentDF)
            val output = model.transform(developmentDF)
            output.printSchema
            output.show()
            output.select("finished_named_entity", "finished_ner_converter").show(false)
            // model.write.overwrite.save(modelPath)
            // evaluate(trainingDF, model, config, "train")
            // evaluate(developmentDF, model, config, "valid")
            // evaluate(testDF, model, config, "test")
          case "predict" =>
            // val model = PipelineModel.load(modelPath)
            // val ef = predict(developmentDF, model, config, "valid")
            // ef.show(false)
          case "eval" => 
            // val model = PipelineModel.load(modelPath)
            // evaluate(trainingDF, model, config, "train")
            // evaluate(developmentDF, model, config, "valid")
            // evaluate(testDF, model, config, "test")
        }

        sc.stop()
      case None => {}
    }

  }
}