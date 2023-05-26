package vlp.llm

import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions._

/**
 * OSCAR-2301 document extractor.
 * <p/>
 * phuonglh, May 24, 2023
 *
 */
object OSCAR {
  def main(args: Array[String]): Unit = {
    val pathInp = if (args.size > 0) args(0) else "/mnt/nfs/dataset/OSCAR/23"
    val pathOut = if (args.size > 1) args(1) else pathInp
    val spark = SparkSession.builder().master("local").appName("OSCAR").getOrCreate()
    val df = spark.read.option("inferSchema", "true").json(pathInp).select("content")
    // filter all documents having more than 80 tokens
    val tokenizer = new Tokenizer().setInputCol("content").setOutputCol("tokens")
    val ef = tokenizer.transform(df)
    val ff = ef.withColumn("size", size(col("tokens"))).filter(col("size") >= 80)
    ff.select("content").repartition(20).write.mode(SaveMode.Overwrite).json(pathOut)
    spark.stop()
  }
}
