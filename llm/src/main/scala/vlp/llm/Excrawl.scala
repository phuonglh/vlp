package vlp.llm

import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import scopt.OptionParser

/**
 * phuonglh9@fpt.com
 *
 * Preprocessor of the excrawl datasets.
 */
object Excrawl {
  def main(args: Array[String]): Unit = {

    val opts = new OptionParser[Config](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory")
      opt[String]('E', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory")
      opt[String]('v', "version").action((x, conf) => conf.copy(version = x)).text("version 23/21/c4")
      opt[String]('i', "inputPath").action((x, conf) => conf.copy(inputPath = x)).text("input path (file/folder)")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path (file/folder)")
      opt[String]('t', "tempPath").action((x, conf) => conf.copy(tempPath = x)).text("temporary directory of Spark")
      opt[Int]('n', "numPartitions").action((x, conf) => conf.copy(numPartitions = x)).text("number of partitions for output")
      opt[String]('l', "level").action((x, conf) => conf.copy(level = x)).text("level to operate, either d (document) or p (paragraph)")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val spark = SparkSession.builder().master(config.master)
          .config("spark.driver.memory", config.driverMemory)
          .config("spark.executor.memory", config.executorMemory)
          .config("spark.executor.cores", config.executorCores)
          .config("spark.deploy.defaultCores", config.totalCores)
          .config("spark.local.dir", config.tempPath)
          .appName("Excrawl").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        val cf = spark.read.options(Map("recursiveFileLookup" -> "true", "pathGlobFilter" -> "*.text")).json(config.inputPath).select("text")
        println(s"Number of input documents = ${cf.count()}")
        val df = cf.distinct()
        println(s"Unique d-level documents = ${df.count()}")
        val ef = OSCAR.distinctParagraph(spark, df, "text")
        println(s"Unique p-level documents  = ${ef.count()}")
        val ff = OSCAR.filterBadRows(ef, "text")
        println(s"Good p-level documents = ${ff.count()}")
        ff.repartition(config.numPartitions).write.option("compression", "gzip").mode(SaveMode.Overwrite).json(config.outputPath)
        spark.stop()
      case None => println("Invalid config.")
    }
  }

}

// -- -i /mnt/data/excrawl/forum -o pre/excrawl/forum -n 1