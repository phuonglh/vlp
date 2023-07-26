package vlp.llm

import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions.{not, _}
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
   executorMemory: String = "8g", // E
   driverMemory: String = "8g", // D
   version: String = "23",
   inputPath: String = "dat/23",
   outputPath: String = "pre/23"
)

object OSCAR {

  def f21(spark: SparkSession, cf: DataFrame): DataFrame = {
    val pairRDD = cf.rdd.zipWithIndex()
    val n = cf.count()
    val emptyLineIdx = pairRDD.map { case (row, id) =>
      val line = row.getAs[String](0).trim
      (line.size, id)
    }.filter(_._1 == 0).map(_._2.toInt).collect() ++ Array(n.toInt)
    // fill an index array for grouping consecutive lines into a document
    val ids = Array.fill[Int](n.toInt)(0)
    var start = 0
    for (i <- 0 until emptyLineIdx.size) {
      val k = emptyLineIdx(i)
      (start until k).foreach(j => ids(j) = i)
      start = k
    }
    import spark.implicits._
    val pairDF = pairRDD.map(p => (p._1.getAs[String](0), p._2)).toDF("content", "lineIdx")
      .filter(length(trim(col("content"))) > 0)
    // zip the pairRDD with ids
    val idsRDD = spark.sparkContext.parallelize(ids).zipWithIndex()
    val idsDF = idsRDD.toDF("docIdx", "lineIdx")
    val df = pairDF.join(idsDF, "lineIdx")
      .withColumn("pair", concat_ws(";", col("lineIdx"), col("content")))
    val ef = df.groupBy("docIdx").agg(collect_list(col("pair")).alias("pairs"))
      .withColumn("xs", array_sort(col("pairs")))
    // sort a list of index-prepended sentences by their index
    // use parallel processing for speed-up
    val sorter = udf((xs: Seq[String]) => {
      xs.par.map { x =>
        val j = x.indexOf(";")
        (x.substring(0, j).toInt, x.substring(j+1))
      }.toList.sortBy(_._1).map(_._2)
    })
    val ff = ef.withColumn("ys", sorter(col("xs")))
      .withColumn("text", concat_ws("\n", col("ys")))
    ff.select("text").filter(not(col("text").contains("sex"))).distinct()
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
    import spark.implicits._
    ffUnique.map { row =>
      row.getAs[String]("content").split("""\n+""")
        .filter(line => line.size >= 40 && line.size <= 2048).filter(_.trim.nonEmpty).mkString("\n")
    }.toDF("text")
  }

  def main(args: Array[String]): Unit = {

    val opts = new OptionParser[Config](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory")
      opt[String]('E', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory")
      opt[String]('v', "version").action((x, conf) => conf.copy(version = x)).text("version 23/22/21")
      opt[String]('i', "inputPath").action((x, conf) => conf.copy(inputPath = x)).text("input path (file/folder)")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path (file/folder)")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        val spark = SparkSession.builder().master("local[*]")
          .config("spark.driver.memory", config.driverMemory)
          .config("spark.executor.memory", config.executorMemory)
          .config("spark.executor.cores", config.executorCores)
          .config("spark.deploy.defaultCores", config.totalCores)
          .appName("OSCAR").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        config.version match {
          case "23" => // same for 22
            val cf = spark.read.option("inferSchema", true).option("recursiveFileLookup", true).json(config.inputPath).select("content")
            val gf = f22(spark, cf)
            gf.select ("text").repartition (10).write.mode (SaveMode.Overwrite).json(config.outputPath)
          case "21" =>
            val cf = spark.read.option("recursiveFileLookup", true).text(config.inputPath).toDF("content")
            val df = f21(spark, cf)
            df.select ("text").repartition (10).write.mode (SaveMode.Overwrite).json(config.outputPath)
            println(s"There are ${df.count()} documents.")
          case "c4" =>
            val df = spark.read.option("recursiveFileLookup", true).json(config.inputPath).select("text")
            val ef = df.distinct()
            ef.repartition(10).write.mode(SaveMode.Overwrite).json(config.outputPath)
            println(s"There are ${ef.count()} documents.")
          case _ =>
            println("Require a version: [23, 21, c4]")
        }
        spark.stop()
      case _ => println("Invalid options")
    }
  }
}
