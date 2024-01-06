package vlp.dep

import com.intel.analytics.bigdl.dllib.utils.Engine
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{DependencyParserModel, PerceptronModel, SentenceDetector, Tokenizer, TypedDependencyParserApproach}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import scala.io.Source
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.ml.Pipeline


case class ConfigDEP(
    master: String = "local[*]",
    totalCores: Int = 8,    // X
    executorCores: Int = 8, // Y ==> there are X/Y executors
    executorMemory: String = "8g", // Z
    driverMemory: String = "16g", // D
    mode: String = "eval",
    batchSize: Int = 128,
    maxSeqLen: Int = 80,
    hiddenSize: Int = 64,
    epochs: Int = 30,
    learningRate: Double = 5E-4,
    modelPath: String = "bin/dep/",
    trainPath: String = "dat/dep/eng/en_ewt-ud-dev.conllu",
    validPath: String = "dat/med/val/", // Parquet file of devPath
    outputPath: String = "out/med/",
    scorePath: String = "dat/med/scores-med.json",
    modelType: String = "s",
)

object DEP {
  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[ConfigDEP](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 16g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('h', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("encoder hidden size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 5E-4")
      opt[String]('d', "trainPath").action((x, conf) => conf.copy(trainPath = x)).text("training data directory")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[String]('s', "scorePath").action((x, conf) => conf.copy(scorePath = x)).text("score path")
    }
    opts.parse(args, ConfigDEP()) match {
      case Some(config) =>
        val conf = new SparkConf().setAppName(getClass.getName).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        Engine.init
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        sc.setLogLevel("ERROR")
        import spark.implicits._

        val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
        val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
        val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")
        val posTagger = PerceptronModel.pretrained().setInputCols("sentence", "token").setOutputCol("pos")
        val dependencyParser = DependencyParserModel.pretrained().setInputCols("sentence", "pos", "token").setOutputCol("dependency")
        val typedDependencyParser = new TypedDependencyParserApproach()
          .setInputCols("dependency", "pos", "token")
          .setOutputCol("dependency_type")
          .setConllU(config.trainPath)
          .setNumberOfIterations(1)

        val pipeline = new Pipeline().setStages(Array(
          documentAssembler,
          sentence,
          tokenizer,
          posTagger,
          dependencyParser,
          typedDependencyParser
        ))

        // Additional training data is not needed, the dependency parser relies on CoNLL-U only.
        val emptyDataSet = Seq.empty[String].toDF("text")
        val pipelineModel = pipeline.fit(emptyDataSet)
        val df = pipelineModel.transform(emptyDataSet)
        pipelineModel.save(config.modelPath)
        df.show()
        spark.stop()
      case None =>

    }
  }
}
