package vlp.con

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.keras.models.Models
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.keras.objectives.SparseCategoricalCrossEntropy
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
 import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer

import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.Row
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.log4j.{Logger, Level}
import com.intel.analytics.bigdl.dllib.keras.objectives.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.optim.Top1Accuracy

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser
import org.slf4j.LoggerFactory


object VSC {
  
 def tokenModel(vocabSize: Int, inputShape: Shape, labelSize: Int, config: Config): Sequential[Float] = {
    val model = Sequential()
    // input to an embedding layer is an index vector of `maxSeqquenceLength` elements, each index is in [0, vocabSize)
    // this layer produces a real-valued matrix of shape `maxSequenceLength x embeddingSize`
    model.add(Embedding(inputDim = vocabSize, outputDim = embeddingSize, inputLength=config.maxSequenceLength))
    // take the matrix above and feed to a GRU layer 
    // by default, the GRU layer produces a real-valued vector of length `recurrentSize` (the last output of the recurrent cell)
    // but since we want sequence information, we make it return a sequences, so the output will be a matrix of shape 
    // `maxSequenceLength x recurrentSize` 
    model.add(GRU(outputDim = config.recurrentSize, returnSequences = true))
    // feed the output of the GRU to a dense layer with relu activation function
    model.add(TimeDistributed(
      Dense(config.hiddenSize, activation="relu").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
      inputShape=Shape(config.maxSequenceLength, config.recurrentSize))
    )
    // add a dropout layer for regularization
    model.add(Dropout(config.dropoutProbability))
    // add the last layer for multi-class classification
    model.add(TimeDistributed(
      Dense(labelSize, activation="softmax").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
      inputShape=Shape(config.maxSequenceLength, config.hiddenSize))
    )
    return model
  }

  def readData(sc: SparkContext, config: Config): Dataset = {
    val df = sc.textFile
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
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Int]('j', "layers").action((x, conf) => conf.copy(layers = x)).text("number of layers, default is 1")
      opt[Int]('r', "recurrentSize").action((x, conf) => conf.copy(recurrentSize = x)).text("number of hidden units in each recurrent layer")
      opt[Int]('h', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("number of hidden units in the dense layer")
      opt[Double]('n', "percentage").action((x, conf) => conf.copy(percentage = x)).text("percentage of the data set to use, default is 0.5")
      opt[Double]('u', "dropoutProbability").action((x, conf) => conf.copy(dropoutProbability = x)).text("dropout ratio, default is 0.2")
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

        val df = None
        df.printSchema()
        df.show()
        // train/test split
        val Array(trainingDF, validationDF) = df.randomSplit(Array(0.8, 0.2), seed = 80L)
        val vocabSize = 20
        val labelSize = 4
        val model = tokenModel(vocabSize, Shape(config.maxSequenceLength), labelSize, config)
        model.compile(optimizer = new Adam(), loss = SparseCategoricalCrossEntropy(), metrics = List(new Top1Accuracy()))

        val transformers = None
        model.fit(trainingDF, batchSize = config.batchSize, nbEpoch = 1, labelCols = Array("label"), valX = validationDF)

        sc.stop()
      case None => {}
    }
  }
}
