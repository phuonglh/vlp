package vlp.vcb

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, isnull, lead, not, to_date, element_at}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.models.{KerasNet, Models}
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.nn.MSECriterion
import com.intel.analytics.bigdl.dllib.utils.{Engine, Shape}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.nnframes.{NNEstimator, NNModel}
import com.intel.analytics.bigdl.dllib.optim.{Adam, Trigger}
import com.intel.analytics.bigdl.dllib.keras.metrics.{MAE, MSE}
import org.apache.spark.SparkContext
import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

case class ConfigStock(
  lead: Int = 10,
  hiddenSize: Int = 64,
  batchSize: Int = 32,
  learningRate: Float = 1E-3f,
  epochs: Int = 10,
  master: String = "local[*]",
  executorCores: String = "4",
  totalCores: String = "8",
  executorMemory: String = "4g",
  driverMemory: String = "4g",
  dataPath: String = "dat/HOSE_2008_2022_vnstock.csv",
  modelPath: String = "bin",
  stock: String = "VNM",
  mode: String = "eval"
)

/**
 * An LSTM-based model for stock price prediction. We use a fix number of previous dates (called lead) to predict the close price of
 * a given stock. Five input features open, high, low, close and volume are normalized to have mean 0 and deviation 1.
 * The target variable (close price) is scaled by 1000.
 *
 */
object Stock {

  private def preprocess(af: DataFrame, config: ConfigStock): DataFrame = {
    val assembler = new VectorAssembler().setInputCols(Array("Open", "High", "Low", "Close", "Volume")).setOutputCol("xRaw").setHandleInvalid("skip")
    val scalerX = new StandardScaler().setInputCol("xRaw").setOutputCol("x0")
    val pipeline = new Pipeline().setStages(Array(assembler, scalerX))
    val pipelineModel = pipeline.fit(af)
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
    concat.transform(ef).select("x", "yRaw").filter(not(isnull(col("yRaw")))).withColumn("y", col("yRaw")/1000)
  }

  private def createModel(config: ConfigStock): KerasNet[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(config.lead, 5), inputShape = Shape(5*config.lead)))
    model.add(LSTM(outputDim = config.hiddenSize, returnSequences = false))
    model.add(Dense(outputDim = 1))
    model
  }

  private def predict(bigdl: KerasNet[Float], df: DataFrame, config: ConfigStock): DataFrame = {
    val sequential = bigdl.asInstanceOf[Sequential[Float]]
    val model = NNModel(sequential).setFeaturesCol("x").setBatchSize(config.batchSize)
    val ef = model.transform(df)
    // "prediction" column contains 1-element array, we extract that only element to a new column "z"
    val ff = ef.withColumn("z", element_at(col("prediction"), 1))
    ff.select("y", "z").repartition(1)
  }

  def main(args: Array[String]): Unit = {
    val opts = new OptionParser[ConfigStock](getClass.getName) {
      head(getClass.getName, "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[String]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 16g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either {eval, train, predict}")
      opt[String]('s', "stock").action((x, conf) => conf.copy(stock = x)).text("stock symbol")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('h', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("encoder hidden size")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x.toFloat)).text("learning rate")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("training data path")
    }
    opts.parse(args, ConfigStock()) match {
      case Some(config) =>
        implicit val formats: AnyRef with Formats = Serialization.formats(NoTypeHints)
        println(Serialization.writePretty(config))
        val conf = Engine.createSparkConf().setAppName(getClass().getName()).setMaster("local[*]")
          .set("spark.executor.cores", config.executorCores).set("spark.cores.max", config.totalCores)
          .set("spark.executor.memory", config.executorMemory).set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        Engine.init
        val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        // read the data and filter the stock symbol of interest
        val af = spark.read.options(Map("header" -> "true", "inferSchema" -> "true")).csv(config.dataPath).filter(col("Stock") === config.stock)
        af.show()
        println(af.count())
        val df = preprocess(af, config)
        df.show()
        df.printSchema()
        println(s"Number of samples = ${df.count()}")
        val Array(trainingDF, validationDF) = df.randomSplit(Array(0.8, 0.2), 1234)
        config.mode match {
          case "train" =>
            val bigdl = createModel(config)
            val trainingSummary = TrainSummary(appName = config.stock, logDir = "sum")
            val validationSummary = ValidationSummary(appName = config.stock, logDir = "sum")
            val estimator = NNEstimator(bigdl, MSECriterion(), Array(5 * config.lead), Array(1))
            estimator.setLabelCol("y").setFeaturesCol("x")
              .setBatchSize(config.batchSize)
              .setOptimMethod(new Adam(config.learningRate))
              .setMaxEpoch(config.epochs)
              .setTrainSummary(trainingSummary)
              .setValidationSummary(validationSummary)
              .setValidation(Trigger.everyEpoch, validationDF, Array(new MSE(), new MAE()), config.batchSize)
            estimator.fit(trainingDF)
            // save the bigdl model
            bigdl.saveModel(s"${config.modelPath}/${config.stock}.bigdl", overWrite = true)
            println(bigdl.summary())
          case "eval" =>
            val bigdl = Models.loadModel[Float](s"${config.modelPath}/${config.stock}.bigdl")
            println(bigdl.summary())
            // 1. validation set
            val vf = predict(bigdl, validationDF, config)
            vf.write.option("overwrite", true).csv(s"${config.modelPath}/${config.stock}-valid.csv")
            // 2. training set
            val tf = predict(bigdl, trainingDF, config)
            tf.write.option("overwrite", true).csv(s"${config.modelPath}/${config.stock}-train.csv")
          case _ =>
        }
        spark.stop()
      case _ => println("No config. Do nothing.")
    }
  }
}
