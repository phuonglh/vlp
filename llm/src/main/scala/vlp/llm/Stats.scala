package vlp.llm

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
 * Compute some statistics of the OSCAR corpus.
 */
object Stats {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]")
      .config("spark.driver.memory", "16g").appName("OSCAR").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val pathInp = args(0)
    val cf = spark.read.option("inferSchema", true).option("recursiveFileLookup", true).json(pathInp)
    println("Number of documents = " + cf.count())
    import spark.implicits._
    val df = cf.map(row => row.getAs[String](0).split("""\s+""").size).toDF("size")
    val ef = df.agg(sum(col("size")))
    ef.show()
    spark.stop()
  }
}
