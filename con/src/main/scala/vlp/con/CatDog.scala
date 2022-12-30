package vlp.con

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.keras.models.Models
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.numeric.NumericFloat

import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.Row
import com.intel.analytics.bigdl.dllib.nnframes.NNImageReader
import org.apache.hadoop.fs.Path
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.log4j.{Logger, Level}
import com.intel.analytics.bigdl.dllib.feature.image.ImageChannelNormalize
import com.intel.analytics.bigdl.dllib.keras.objectives.BinaryCrossEntropy
import com.intel.analytics.bigdl.dllib.optim.Top1Accuracy
import com.intel.analytics.bigdl.dllib.feature.image.RowToImageFeature
import com.intel.analytics.bigdl.dllib.feature.image.ImageResize
import com.intel.analytics.bigdl.dllib.feature.image.ImageCenterCrop
import com.intel.analytics.bigdl.dllib.feature.image.ImageMatToTensor
import com.intel.analytics.bigdl.dllib.feature.image.ImageFeatureToTensor
import com.intel.analytics.bigdl.dllib.feature.common.ChainedPreprocessing

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser
import org.slf4j.LoggerFactory


object CatDog {
  
 def buildModel(inputShape: Shape): Sequential[Float] = {
    val model = Sequential()
    model.add(Conv2D(32, 3, 3, inputShape = inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Conv2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    return model
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val logger = LoggerFactory.getLogger(getClass.getName)


    val opts = new OptionParser[Config](getClass().getName()) {
      head(getClass().getName(), "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[Int]('X', "executorCores").action((x, conf) => conf.copy(executorCores = x)).text("executor cores, default is 8")
      opt[Int]('Y', "totalCores").action((x, conf) => conf.copy(totalCores = x)).text("total number of cores, default is 8")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('D', "driverMemory").action((x, conf) => conf.copy(driverMemory = x)).text("driver memory, default is 8g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('h', "hiddenUnits").action((x, conf) => conf.copy(hiddenUnits = x)).text("number of hidden units in each layer")
      opt[Int]('j', "layers").action((x, conf) => conf.copy(layers = x)).text("number of layers, default is 1")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Double]('n', "percentage").action((x, conf) => conf.copy(percentage = x)).text("percentage of the data set to use, default is 0.5")
      opt[Double]('u', "dropout").action((x, conf) => conf.copy(dropout = x)).text("dropout ratio, default is 0")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('l', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("max sequence length")
      opt[Double]('a', "alpha").action((x, conf) => conf.copy(learningRate = x)).text("learning rate, default value is 0.001")
      opt[Boolean]('g', "gru").action((x, conf) => conf.copy(gru = x)).text("use 'gru' if true, otherwise use lstm")
      opt[String]('d', "dataPath").action((x, conf) => conf.copy(dataPath = x)).text("data path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model folder, default is 'bin/'")
      opt[String]('i', "inputPath").action((x, conf) => conf.copy(inputPath = x)).text("input data path")
      opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode, default is false")
    }
    opts.parse(args, Config()) match {
      case Some(config) =>
        implicit val formats = Serialization.formats(NoTypeHints)
        println(Serialization.writePretty(config))

        val conf = Engine.createSparkConf().setAppName(getClass().getName()).setMaster(config.master)
          .set("spark.executor.cores", config.executorCores.toString)
          .set("spark.cores.max", config.totalCores.toString)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.driver.memory", config.driverMemory)
        val sc = new SparkContext(conf)
        Engine.init

        // read the "cats_dogs/train" folder and create labeled images: cat -> 1, dog -> 2
        val createLabel = udf { row: Row => 
          if (new Path(row.getString(0)).getName.contains("cat")) 1 else 2 
        }
        val imagePath = "dat/cats_dogs/demo/"
        val imageDF = NNImageReader.readImages(imagePath, sc, resizeH = 256, resizeW = 256)
        val df = imageDF.withColumn("label", createLabel(col("image")))
        df.printSchema()
        df.show()
        // train/test split
        val Array(trainingDF, validationDF) = df.randomSplit(Array(0.8, 0.2), seed = 80L)

        val model = buildModel(Shape(3, 256, 256))
        model.compile(optimizer = new Adam(), loss = BinaryCrossEntropy(), metrics = List(new Top1Accuracy()))

        // preprocess the images before training
        // val transformers = ChainedPreprocessing(Array(RowToImageFeature(), ImageResize(256, 256), 
        //   ImageCenterCrop(224, 224), ImageChannelNormalize(123, 117, 104), ImageMatToTensor(), ImageFeatureToTensor()))
        val transformers = ImageChannelNormalize(123, 117, 104)
        model.fit(trainingDF, batchSize = config.batchSize, nbEpoch = 5, labelCols = Array("label"), 
          transform = transformers, valX = validationDF)

        sc.stop()
      case None => {}
    }
  }
}
