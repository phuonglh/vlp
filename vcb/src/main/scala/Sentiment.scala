package vlp.vcb

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, StringIndexer, Tokenizer}
import org.apache.spark.sql.functions.{col, isnull, not}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Sentiment {

  def createPipeline(df: DataFrame): PipelineModel = {
    val labelIndexer = new StringIndexer().setInputCol("sentiment").setOutputCol("label")
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    val countVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("c").setMinDF(2)
    val idf = new IDF().setInputCol("c").setOutputCol("v")
    val classifier = new LogisticRegression().setFeaturesCol("v").setLabelCol("label")
    val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, countVectorizer, idf, classifier))
    pipeline.fit(df)
  }

  def evaluate(df: DataFrame, model: PipelineModel) = {
    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol("label")
    val score = evaluator.evaluate(df)
    print(s"areaUnderROC = $score\n")
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Sentiment").master("local").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val df = spark.read.option("header", "true").csv("dat/sentiments.csv")
      .filter(not(isnull(col("sentiment"))))
    df.show
    println(s"${df.count} records")
    df.printSchema()

    val Array(trainingDF, testDF) = df.randomSplit(Array(0.8, 0.2), 1234)
    val model = createPipeline(trainingDF)
    val trainingPrediction = model.transform(trainingDF)
    val testPrediction = model.transform(testDF)

    testPrediction.show

    evaluate(trainingPrediction, model)
    evaluate(testPrediction, model)

    spark.stop()

  }
}

// term/word/token ==> count(term)/ term frequency (tf)
// document frequency (df) ==> tf/df (tf.idf)
// count: test areaUnderROC = 0.7420423903968029

// areaUnderROC = 0.766543136369237, minDF = 2, tf-idf