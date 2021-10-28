package vlp.vcb

import scala.math.random
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

/**
  * phuonglh@gmail.com
  * 
  */
object SparkPi {
  def main(args: Array[String]) {
    val spark = SparkSession.builder.appName("vlp.vcb.BankMarkting")
        .master("local[*]")
        .getOrCreate()
    val df = spark.read
        .option("header", "true").option("delimiter", ";").option("inferSchema", "true")
        .csv("dat/bank-additional/bank-additional.csv")
    df.show()
    println(s"Number of records = ${df.count()}")
    df.printSchema()

    df.groupBy(col("age")).sum("duration").show()

    val dayIndexer = new StringIndexer().setInputCol("day_of_week").setOutputCol("day_of_week_index")
    val dayEncoder = new OneHotEncoder().setInputCol("day_of_week_index").setOutputCol("day_of_week_encoded")
    val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
    val assembler = new VectorAssembler().setInputCols(Array("age", "duration", "day_of_week_encoded")).setOutputCol("features")
    val pipeline = new Pipeline().setStages(Array(dayIndexer, dayEncoder, assembler, labelIndexer))
    val pipelineModel = pipeline.fit(df)
    val df1 = pipelineModel.transform(df)

    val df2 = df1.select("age", "duration", "day_of_week", "day_of_week_encoded", "features", "label")
    // cache the training data for speed processing
    df2.cache()
    df2.show()

    val kmeans = new KMeans().setK(10).setSeed(1L)
    val kmeansModel = kmeans.fit(df2)
    val predictions = kmeansModel.transform(df2)
    val evaluator = new ClusteringEvaluator()
    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared Euclidean distance = $silhouette")

    // Shows the result.
    println("Cluster Centers: ")
    kmeansModel.clusterCenters.foreach(println)

    spark.stop()
  }
}
