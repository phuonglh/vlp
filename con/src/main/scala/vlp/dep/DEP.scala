package vlp.dep

import com.intel.analytics.bigdl.dllib.NNContext
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import scala.io.Source
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}


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
    trainPath: String = "dat/dep/eng/2.7/en_ewt-ud-dev.conllu",
    validPath: String = "dat/dep/eng/2.7/en_ewt-ud-test.conllu",
    outputPath: String = "out/dep/",
    scorePath: String = "dat/dep/scores.json",
    modelType: String = "s",
)

object DEP {

  /**
   * Linearize a graph into 4 seqs: Seq[word], Seq[PoS], Seq[labels], Seq[offsets].
   * @param graph
   * @return a sequence of sequences.
   */
  def linearize(graph: Graph): Seq[Seq[String]] = {
    val tokens = graph.sentence.tokens.tail // remove the ROOT token at the beginning
    val words = tokens.map(_.word)
    val partsOfSpeech = tokens.map(_.partOfSpeech)
    val labels = tokens.map(_.dependencyLabel)
    val offsets = tokens.map(token => (token.head.toInt - token.id.toInt).toString) // offset from the head
    Seq(words, partsOfSpeech, labels, offsets)
  }

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
        // Creates or gets SparkContext with optimized configuration for BigDL performance.
        // The method will also initialize the BigDL engine.
        val sc = NNContext.initNNContext(conf)
        sc.setLogLevel("ERROR")
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        import spark.implicits._
        val graphs = GraphReader.read(config.trainPath)
        val xs = graphs.map { graph => Row(linearize(graph):_*) } // need to scroll out the parts with :_*
        val schema = StructType(Array(
          StructField("tokens", ArrayType(StringType, true)),
          StructField("partsOfSpeech", ArrayType(StringType, true)),
          StructField("labels", ArrayType(StringType, true)),
          StructField("offsets", ArrayType(StringType, true))
        ))
        val df = spark.createDataFrame(sc.parallelize(xs), schema)
        println(df.count)
        df.show(5, false)

        spark.stop()
      case None =>

    }
  }
}
