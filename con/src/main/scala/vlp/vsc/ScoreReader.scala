package vlp.vsc

import org.apache.spark.SparkConf
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._

/**
  * Reads scores and performs aggregation for report.
  * 
  * phuonglh@gmail.com
  * 
  */
object ScoreReader {  
  def averageLSTM(spark: SparkSession, scorePath: String) = {
    val df = spark.read.option("multiline", "true").json(scorePath)
    val language = df.select("input").first.getString(0)
    // filter by model
    val modelTypes = Seq("tk", "st", "ch")
    // val modelTypes = Seq("ch")
    for (t <- modelTypes) {
      val ef = df.filter(col("modelType") === t)
      val splits = Seq("train", "valid")
      for (s <- splits) {
        val ff = ef.filter(col("split") === s)
        // the fMeasure field is of array type, we cannot average it (Spark does not support)
        // hence we split it into two numeric columns first
        val bf = ff.select(col("embeddingSize"), col("encoderSize"), col("layerSize"), 
          col("accuracy"), 
          col("precision").getItem(0).as("p0"), col("precision").getItem(1).as("p1"),
          col("recall").getItem(0).as("r0"), col("recall").getItem(1).as("r1"),
          col("fMeasure").getItem(0).as("f0"), col("fMeasure").getItem(1).as("f1")
        )
        // and group the bf by the triple of 3 hyper-params
        val gf = bf.groupBy(col("embeddingSize"), col("encoderSize"), col("layerSize"))
        // and average the accuracy and f-measure scores
        val average = gf.agg(
          avg("accuracy"), stddev("accuracy"), 
          avg("p0"), avg("p1"),
          avg("r0"), avg("r1"),
          avg("f0"), avg("f1")
        ).sort(col("embeddingSize"), col("encoderSize"), col("layerSize"))
        // write out the result        
        average.repartition(1).write.option("header", "true").option("delimiter", "\t").csv(s"dat/vsc/result/$language-$t-$s")
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val scorePath = args(0)
    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[2]")
    val spark = SparkSession.builder.config(conf).getOrCreate()
    averageLSTM(spark, scorePath)
    spark.stop()
  }
}
