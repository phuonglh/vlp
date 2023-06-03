import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}

object CreditRisk {

  def preprocess(df: DataFrame): DataFrame = {
    val df0 = df.filter(col("default_ind") === 0).sample(0.1, 1234)
    val df1 = df.filter(col("default_ind") === 1)
    df0.union(df1).na.fill(0f)
  }

  def createPipeline(df: DataFrame): PipelineModel = {
    val gradeIndexer = new StringIndexer().setInputCol("grade").setOutputCol("gradeIdx")
    val gradeEncoder = new OneHotEncoder().setInputCol("gradeIdx").setOutputCol("gradeVec").setDropLast(false)
    val purposeIndexer = new StringIndexer().setInputCol("purpose").setOutputCol("purposeIdx")
    val purposeEncoder = new OneHotEncoder().setInputCol("purposeIdx").setOutputCol("purposeVec").setDropLast(false)
    // find all double columns
    val doubleCols = df.schema.fields.filter(_.dataType == DoubleType).map(_.name)
    val assembler = new VectorAssembler().setInputCols(Array("gradeVec", "purposeVec") ++ doubleCols).setOutputCol("features")
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("x")
    val classifier = new MultilayerPerceptronClassifier()
      .setLabelCol("default_ind").setBlockSize(32)
      .setFeaturesCol("x")
      .setLayers(Array(21 + doubleCols.size, 16, 2))
    val pipeline = new Pipeline().setStages(Array(gradeIndexer, gradeEncoder, purposeIndexer, purposeEncoder,
        assembler, scaler, classifier))
    val model = pipeline.fit(df)
    model.write.overwrite().save("models/credit")
    val ef = model.transform(df)
    ef.select("gradeVec", "purposeVec", "features", "default_ind", "prediction").show(false)
    model
  }
  def evaluate(df: DataFrame, model: PipelineModel) = {
    val ef = model.transform(df)
    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol("default_ind")
    val score = evaluator.evaluate(ef)
    print(s"areaUnderROC = $score")
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("CreditRisk").master("local")
      .config("executor.memory", "4g")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val df = spark.read.options(Map("inferSchema" -> "true", "delimiter" -> """\t""", "header" -> "true")).csv("dat/lending.txt.gz")
    df.show
    println(s"${df.count} records")
    df.printSchema()

    val af = preprocess(df)
    println(s"${af.count} records")

    val model = createPipeline(af)
    evaluate(af, model)

    spark.stop()
  }
}
