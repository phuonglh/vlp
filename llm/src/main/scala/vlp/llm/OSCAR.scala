package vlp.llm

import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import scopt.OptionParser

/**
 * OSCAR-2301 document extractor.
 * <p/>
 * This utility reads OSCAR-23 json file(s) into a data frame, tokenizes the content field, filters all documents
 * containing less than 80 syllables, flat map the contents into lines, filters line of length more than 40 characters
 * and less than 2048 characters; and writes the result into a text directory (20 partitions).
 *
 *
 * <p/>
 * phuonglh, May 24, 2023
 *
 */

case class Config(
   master: String = "local[*]",
   totalCores: Int = 8, // X
   executorCores: Int = 8, // Y ==> there are Y/X executors
   executorMemory: String = "8g", // Z
   driverMemory: String = "8g", // D
   version: String = "23",
   inputPath: String = "dat/23",
   outputPath: String = "pre/23"
)

object OSCAR {

  def f21(spark: SparkSession, cf: DataFrame): Unit = {
    
  }

  def f22(spark: SparkSession, cf: DataFrame): DataFrame = {
    // filter all documents containing toxic contents: "sex"
    val df = cf.filter(not(col("content").contains("sex")))
    // filter all documents having more than 80 tokens
    val tokenizer = new Tokenizer().setInputCol("content").setOutputCol("tokens")
    val ef = tokenizer.transform(df)
    val ff = ef.withColumn("size", size(col("tokens"))).filter(col("size") >= 80)
    print(s"Number of filtered documents = ${ff.count}\n")
    // dedup the document (whole document level)
    val ffUnique = ff.select("content").distinct()
    // split the documents by the new-line character
    import spark.implicits._
    ffUnique.map { row =>
      row.getAs[String]("content").split("""\n+""")
        .filter(line => line.size >= 40 && line.size <= 2048)
    }.toDF("lines").filter(row => row.getAs[Seq[String]](0).size > 0)
  }

  def main(args: Array[String]): Unit = {

    val opts = new OptionParser[Config](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores")
      opt[String]('v', "version").action((x, conf) => conf.copy(version = x)).text("version 23/22/21")
      opt[String]('i', "inputPath").action((x, conf) => conf.copy(inputPath = x)).text("input path (file/folder)")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path (file/folder)")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val spark = SparkSession.builder().master("local[*]")
          .config("spark.driver.memory", config.driverMemory)
          .appName("OSCAR").getOrCreate()
        config.version match {
          case "23" =>
            val cf = spark.read.option("inferSchema", true).option("recursiveFileLookup", true).json(config.inputPath).select("content")
            val gf = f22(spark, cf)
            gf.select ("lines").repartition (10).write.mode (SaveMode.Overwrite).json(config.outputPath)
          case "21" =>
            val cf = spark.read.option("recursiveFileLookup", true).text(config.inputPath).toDF("content")
            println(cf.count)
            val df = cf.filter(length(trim(col("content"))) === 0)
            println(df.count)
          case _ =>
        }
        spark.stop()
      case _ => println("Invalid options")
    }
  }
}
