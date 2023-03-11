package vlp.woz.act

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
    // filter by model
    val ef = df.filter(col("modelType") === "lstm")
    val splits = Seq("train", "valid", "test")
    for (s <- splits) {
      val ff = ef.filter(col("split") === s)
      val bf = ff.select(col("layerSize"), col("embeddingSize"), col("encoderSize"), col("accuracy"), col("f1Measure"), col("microF1Measure"))
      // and group the bf by the triple of 3 hyper-params
      val gf = bf.groupBy(col("embeddingSize"), col("encoderSize"), col("layerSize"))
      // and average the accuracy and f-measure scores
      val average = gf.agg(round(avg("accuracy"), 4).alias(s"accuracy-$s"), round(avg("f1Measure"), 4).alias(s"f1Measure-$s"), round(avg("microF1Measure"), 4).alias(s"microF1Measure-$s")).sort(col("layerSize"), col("embeddingSize"), col("encoderSize"))
      // write out the result        
      average.repartition(1).write.option("header", "true").option("delimiter", "\t").csv(s"dat/woz/out/lstm-$s")
    }
  }

  def main(args: Array[String]): Unit = {
    val scorePath = args(0)
    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[2]")
    val spark = SparkSession.builder.config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    averageLSTM(spark, scorePath)
    spark.stop()
  }
}
