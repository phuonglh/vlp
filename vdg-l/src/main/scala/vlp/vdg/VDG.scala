package vlp.vdg

import java.nio.file.{Files, Paths, StandardOpenOption}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.utils.serializer.ModuleLoader

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame
import org.json4s._
import org.json4s.jackson.Serialization
import org.slf4j.LoggerFactory
import scopt.OptionParser
import com.intel.analytics.bigdl.mkl.MKL
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.RowFactory

/**
  * Vietnamese Diacritics Generation
  *
  * On Spark good configuration. Suppose that we 6 nodes, each node has 20 cores and 64g of RAM.
  *
  * Option 1:
  *
  * X: --executor-cores. This option specifies that each executor can run a maximum of X tasks at the same time.
  * Y: --total-executor-cores. This is the total number of cores. In our example, we have a total of 6x20 = 120 cores.
  *
  * If we set Y be 108, and X be 6, then we have the 108/6 = 18 executors (--num-executors=18). That is, on each node,
  * we have 18/6 = 3 executors.
  *
  * If on each node, 63g of RAM is reserved for Spark computation (1g for other services), each executor can use 63/3 = 21g
  * of RAM for computation. A better parameter for this should be 19g. That is, we should set --executor-memory=19g.
  *
  * spark-submit --executor-memory 19g --driver-memory 60g .../vdg.jar -Y 108 -X 6
  *
  * The batch size needs to divide both 6 and 18. We can choose -b=216, which is (3x6)x4. This will put 4 samples for each
  * running core.
  *
  * Option 2:
  * X = 10, Y = 60, then there are 6 executors. The executor memory = 4g. The batch size should be 4x10 = 40.
  *
  */
object VDG {
  final val logger = LoggerFactory.getLogger(VDG.getClass.getName)
  final val partition = Array(0.8, 0.2)

  def eval(config: ConfigVDG, vdg: M, dataSet: DataFrame, preprocessor: PipelineModel, module: Module[Float], trainingTime: Long = 0): Unit = {
    val Array(trainingSet, validationSet) = dataSet.randomSplit(partition, 150909L)
    logger.info("#(samples) = " + dataSet.count())
    logger.info("#(trainingSamples) = " + trainingSet.count())
    logger.info("#(validationSamples) = " + validationSet.count())
    val numTrainingSamples = (trainingSet.count() / config.batchSize) * config.batchSize
    val numValidationSamples = (validationSet.count() / config.batchSize) * config.batchSize
    val trainingScore = vdg.eval(trainingSet.limit(numTrainingSamples.toInt), preprocessor, module)
    val validationScore = vdg.eval(validationSet.limit(numValidationSamples.toInt), preprocessor, module)
    val eval = ConfigEval("vdg", config.dataPath, config.percentage, config.modelPath, config.modelType,
      if (config.gru) "GRU"; else "LSTM", config.layers, config.hiddenUnits, trainingScore, validationScore, validationScore, trainingTime)
    implicit val formats = Serialization.formats(NoTypeHints)
    val content = Serialization.writePretty(eval) + ",\n"
    Files.write(Paths.get(config.logPath), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val parser = new OptionParser[ConfigVDG]("VDG") {
      head("vlp.vdg", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 8g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('h', "hiddenUnits").action((x, conf) => conf.copy(hiddenUnits = x)).text("number of hidden units in each layer")
      opt[Int]('j', "layers").action((x, conf) => conf.copy(layers = x)).text("number of layers, default is 1")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs, default is 100")
      opt[Double]('n', "percentage").action((x, conf) => conf.copy(percentage = x)).text("percentage of the data set to use, default is 0.5")
      opt[Double]('u', "dropout").action((x, conf) => conf.copy(dropout = x)).text("dropout ratio, default is 0")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('l', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("max sequence length in chars or tokens, depending on model type")
      opt[Int]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type to use, from 1 to 4")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 0.001")
      opt[Boolean]('g', "gru").action((x, conf) => conf.copy(gru = x)).text("use 'gru' if true, otherwise use lstm")
      opt[Unit]('q', "peephole").action((x, conf) => conf.copy(peephole = true)).text("use 'peephole' connection with LSTM")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("data path, default is 'dat/hcm.txt'")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('i', "inputPath").action((x, conf) => conf.copy(inputPath = x)).text("input path, default is 'dat/hcm.txt'")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path, default is 'dat/hcm.out'")
      opt[Unit]('y', "jsonData").action((x, conf) => conf.copy(jsonData = true)).text("use JSON dataset, default is true for 'dat/txt/news.json'")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode, default is false")
    }
    parser.parse(args, ConfigVDG()) match {
      case Some(config) =>
        implicit val formats = Serialization.formats(NoTypeHints)
        logger.info(Serialization.writePretty(config))
        val vdg = new M1(config) 
        val conf = Engine.createSparkConf()
          .setAppName("VDG")
          .setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
          .set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
          .set("spark.executor.extraJavaOptions", "-Dcom.github.fommil.netlib.BLAS=com.intel.mkl.MKLBLAS -Dcom.github.fommil.netlib.LAPACK=com.intel.mkl.MKLLAPACK")
          .set("spark.executorEnv.MKL_VERBOSE", "1")
        val sparkContext = new SparkContext(conf)
        Engine.init

        // use MKL to speedup the processing
        MKL.setNumThreads(4)
    
        val modelSt = "M" + config.modelType + (if (config.gru) "G"; else "L") + config.layers + "H" + config.hiddenUnits
        // create a path to store this model using a given base modelPath
        val path = config.modelPath + (if (config.modelPath.endsWith("/")) "" else "/") + s"${modelSt}/"

        val needDataModes = Set("train", "eval", "exp")
        if (needDataModes.contains(config.mode)) {
          val dataSet = if (config.jsonData)
            IO.readJsonFiles(sparkContext, config.dataPath).sample(config.percentage, 220712L)
          else IO.readTextFiles(sparkContext, config.dataPath).sample(config.percentage, 220712L)
          // add additional data files
          val additionalDataSet = IO.readTextFiles(sparkContext, "dat/hcm-addition.txt")

          val Array(trainingSet, validationSet) = dataSet.randomSplit(partition, 150909L)
          logger.info("#(samples) = " + dataSet.count())
          val trainingSet2 = trainingSet.union(additionalDataSet)
          logger.info("#(trainingSamples) = " + trainingSet2.count())
          logger.info("#(validationSamples) = " + validationSet.count())
          val numValidationSamples = (validationSet.count() / config.batchSize) * config.batchSize

          config.mode match {
            case "train" =>
              val startTime = System.currentTimeMillis()
              val module = vdg.train(trainingSet2, validationSet.limit(numValidationSamples.toInt))
              val endTime = System.currentTimeMillis()
              val trainingTime = (endTime - startTime)/1000
              val preprocessor = PipelineModel.load(path)
              eval(config, vdg, validationSet, preprocessor, module, trainingTime)
            case "eval" =>
              val preprocessor = PipelineModel.load(path)
              val module = ModuleLoader.loadFromFile[Float](path + "vdg.bigdl", path + "vdg.bin")
              eval(config, vdg, validationSet, preprocessor, module)
            case "exp" =>
              for (m <- 1 to 3) {
                val startTime = System.currentTimeMillis()
                val module = vdg.train(trainingSet, validationSet.limit(numValidationSamples.toInt))
                val endTime = System.currentTimeMillis()
                val trainingTime = (endTime - startTime)/1000
                val preprocessor = PipelineModel.load(path)
                eval(config, vdg, dataSet, preprocessor, module, trainingTime)
              }
          }
        } else {
          config.mode match {
            case "predict" =>
              val input = IO.readTextFiles(sparkContext, config.inputPath)
              val preprocessor = PipelineModel.load(path)
              val module = ModuleLoader.loadFromFile[Float](path + "vdg.bigdl", path + "vdg.bin")
              logger.info(module.toString)
              val rdd = vdg.predict(input, preprocessor, module).map { row => 
                RowFactory.create(row.getAs[Seq[String]](0).mkString, row.getAs[Seq[String]](1).mkString, row.getAs[Seq[String]](2).mkString)
              }
              val sparkSession = SparkSession.builder().getOrCreate()
              import sparkSession.implicits._
              val schema = StructType(Array(StructField("x", StringType, false), StructField("y", StringType, false), StructField("z", StringType, false)))
              val df = sparkSession.createDataFrame(rdd, schema)
              df.write.json(config.outputPath)
            case "run" =>
              val preprocessor = PipelineModel.load(path)
              val module = ModuleLoader.loadFromFile[Float](path + "vdg.bigdl", path + "vdg.bin")
              var text = ""
              logger.info("Enter a non-accent text. Enter an empty line (Enter) to quit.")
              do {
                text = scala.io.StdIn.readLine()
                println(s"""You entered: "$text" """)
                if (text.trim.nonEmpty) {
                  val output = vdg.test(text, preprocessor, module)
                  logger.info(output)
                }
              } while (text.trim.nonEmpty)
          }
        }
      case None => {}
    }
  }
}
