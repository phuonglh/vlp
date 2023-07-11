package vlp.llm

import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions._

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
object OSCAR {
  def main(args: Array[String]): Unit = {
    val pathInp = if (args.size > 0) args(0) else "dat/23"
    val pathOut = if (args.size > 1) args(1) else "pre/23"
    val spark = SparkSession.builder().master("local[*]").config("spark.driver.memory", "12g").appName("OSCAR").getOrCreate()
    val df = spark.read.option("inferSchema", "true").json(pathInp).select("content")
    // filter all documents having more than 80 tokens
    val tokenizer = new Tokenizer().setInputCol("content").setOutputCol("tokens")
    val ef = tokenizer.transform(df)
    val ff = ef.withColumn("size", size(col("tokens"))).filter(col("size") >= 80)
    print(s"Number of filtered documents = ${ff.count}\n")
    // split the documents by the new-line character
    import spark.implicits._
    val gf = ff.select("content").flatMap(row => row.getAs[String]("content").split("""\n+""")).toDF("line")
    // filter lines based on their length
    val hf = gf.withColumn("length", length(col("line"))).filter(col("length") >= 40 && col("length") <= 2048)
    print(s"Number of filtered lines = ${hf.count}\n")
    hf.select("line").distinct().repartition(10).write.mode(SaveMode.Overwrite).text(pathOut)
    spark.stop()
  }
}
