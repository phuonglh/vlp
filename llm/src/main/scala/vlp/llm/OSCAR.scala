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
    val pathInp = if (args.size > 0) args(0) else "dat/23/9"
    val pathOut = if (args.size > 1) args(1) else "pre/23/9"
    val spark = SparkSession.builder().master("local[*]").config("spark.driver.memory", "64g").appName("OSCAR").getOrCreate()
    val cf = spark.read.option("inferSchema", true).option("recursiveFileLookup", true).json(pathInp).select("content")
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
    val gf = ffUnique.map { row =>
      row.getAs[String]("content").split("""\n+""")
        .filter(line => line.size >= 40 && line.size <= 2048)
    }.toDF("lines").filter(row => row.getAs[Seq[String]](0).size > 0)
    gf.select("lines").repartition(10).write.mode(SaveMode.Overwrite).json(pathOut)
    spark.stop()
  }
}
