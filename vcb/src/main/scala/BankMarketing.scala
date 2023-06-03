import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object BankMarketing  {

  def createPipeline(df: DataFrame): PipelineModel = {
    val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIdx")
    val jobEncoder = new OneHotEncoder().setInputCol("jobIdx").setOutputCol("jobVec").setDropLast(false)
    val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIdx")
    val maritalEncoder = new OneHotEncoder().setInputCol("maritalIdx").setOutputCol("maritalVec").setDropLast(false)
    val assembler = new VectorAssembler().setInputCols(Array("age", "jobVec", "maritalVec", "duration", "euribor3m")).setOutputCol("features")

    val yIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
    val classifier = new LogisticRegression()

    val pipeline = new Pipeline().setStages(Array(jobIndexer, jobEncoder, maritalIndexer, maritalEncoder, assembler, yIndexer, classifier))
    pipeline.fit(df)
  }


  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("BankMarketing").getOrCreate()
    val df = spark.read
      .options(Map("inferSchema" -> "true", "delimiter" -> ";", "header" -> "true"))
      .csv("dat/marketing.csv")
    df.show
    df.printSchema()

    // create and apply a processing pipeline
    val pipelineModel = createPipeline(df)
    val ef = pipelineModel.transform(df)
    ef.select("features", "label", "prediction").show(50, false)

//    ef.groupBy("label").count().show(false)
//    ef.groupBy("prediction").count().show(false)
    val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
    val score = evaluator.evaluate(ef)
    println(s"training score = $score")

    spark.stop()
  }

}
