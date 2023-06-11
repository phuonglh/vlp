import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, isnull, lead, not, to_date}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.models.KerasNet
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.nn.MSECriterion
import com.intel.analytics.bigdl.dllib.utils.{Engine, Shape}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.nnframes.{NNEstimator, NNModel}
import com.intel.analytics.bigdl.dllib.optim.{Adam, Trigger}
import com.intel.analytics.bigdl.dllib.keras.metrics.{MSE, MAE}

import org.apache.spark.SparkContext

case class ConfigStock(
  lead: Int = 10,
  recurrentSize: Int = 64,
  batchSize: Int = 32,
  learningRate: Float = 1E-3f,
  epochs: Int = 40,
  executorCores: String = "4",
  totalCores: String = "8",
  executorMemory: String = "4g",
  driverMemory: String = "4g"
)

object Stock {

  private def preprocess(af: DataFrame, config: ConfigStock): DataFrame = {
    val assembler = new VectorAssembler().setInputCols(Array("Open", "High", "Low", "Close", "Volume")).setOutputCol("xRaw").setHandleInvalid("skip")
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
    val concat = new VectorAssembler().setInputCols(colNames).setOutputCol("x").setHandleInvalid("skip")
    concat.transform(ef).select("x", "yRaw")
      .filter(not(isnull(col("yRaw"))))
      .withColumn("y", col("yRaw")/1000)
  }

  def createModel(config: ConfigStock): KerasNet[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(config.lead, 5), inputShape = Shape(5*config.lead)))
    model.add(LSTM(outputDim = config.recurrentSize, returnSequences = false))
    model.add(Dense(outputDim = 1))
    model
  }

  def linearRegression(df: DataFrame) = {
    val lr = new LinearRegression().setFeaturesCol("x").setLabelCol("y")
    val lrModel = lr.fit(df)
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }

  def main(args: Array[String]): Unit = {
    val config = ConfigStock()
    val conf = Engine.createSparkConf().setAppName(getClass().getName()).setMaster("local[*]")
      .set("spark.executor.cores", config.executorCores)
      .set("spark.cores.max", config.totalCores)
      .set("spark.executor.memory", config.executorMemory)
      .set("spark.driver.memory", config.driverMemory)
    val sc = new SparkContext(conf)
    Engine.init

    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val af = spark.read.options(Map("header" -> "true", "inferSchema" -> "true"))
      .csv("dat/HOSE_2008_2022_vnstock.csv")
      .filter(col("Stock") === "VNM")
    af.show(false)
    println(af.count())
    val df = preprocess(af, config)
    df.show(false)
    df.printSchema()
    println(df.count())

    val bigdl = createModel(config)
    val trainingSummary = TrainSummary(appName = "stock", logDir = "sum")
    val validationSummary = ValidationSummary(appName = "stock", logDir = "sum")
    val estimator = NNEstimator(bigdl, MSECriterion(), Array(5*config.lead), Array(1))
    estimator.setLabelCol("y").setFeaturesCol("x")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(config.learningRate))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainingSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, df, Array(new MSE(), new MAE()), config.batchSize)
    estimator.fit(df)

    spark.stop
  }
}
