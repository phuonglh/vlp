import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, lead, to_date}

case class ConfigStock(
  val lead: Int = 5
)

object Stock {

  private def preprocess(af: DataFrame, config: ConfigStock): DataFrame = {
    val assembler = new VectorAssembler().setInputCols(Array("Open", "High", "Low", "Close", "Volume")).setOutputCol("xRaw")
    val scalerX = new StandardScaler().setInputCol("xRaw").setOutputCol("x0")
    val pipeline = new Pipeline().setStages(Array(assembler, scalerX))
    val pipelineModel = pipeline.fit(af) // TODO: save params for later re-scaling
    val bf = pipelineModel.transform(af)
    // add "Date" column
    val cf = bf.withColumn("Date", to_date(col("TradingDate"), "yyy-MM-dd"))
    // define a window over the stock symbol and order by date
    val window = Window.partitionBy("Stock").orderBy(col("Date"))
    val df = cf.withColumn("yRaw", lead("Close", config.lead).over(window))
    var ef = df
    for (k <- 1 until config.lead) {
      ef = ef.withColumn(s"x$k", lead("x0", k).over(window))
    }
    df.show(false)
    val colNames = (0 until config.lead).map(k => s"x$k").toArray
    val concat = new VectorAssembler().setInputCols(colNames).setOutputCol("x")
    concat.transform(ef).select("x", "yRaw")
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Stock").master("local").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val af = spark.read.options(Map("header" -> "true", "inferSchema" -> "true"))
      .csv("dat/HOSE_2008_2022_vnstock.csv")
      .filter(col("Stock") === "VNM")
    af.show(false)
    println(af.count())

    val df = preprocess(af, ConfigStock())
    df.show(false)
    df.printSchema()

    spark.stop
  }
}
