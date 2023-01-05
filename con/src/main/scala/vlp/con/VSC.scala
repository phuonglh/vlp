package vlp.con

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.keras.models.Models
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.TimeDistributedCriterion
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.optim.Loss
import com.intel.analytics.bigdl.dllib.utils.Engine

import org.apache.spark.SparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature._
import org.apache.spark.ml._

import org.json4s._
import org.json4s.jackson.Serialization
import scopt.OptionParser

import org.apache.log4j.{Logger, Level}
import org.slf4j.LoggerFactory
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.optim.Trigger
import com.intel.analytics.bigdl.dllib.nn.{TimeDistributedCriterion, ClassNLLCriterion}


object VSC {
  
 def tokenModel(vocabSize: Int, labelSize: Int, inputShape: Shape, config: Config): Sequential[Float] = {
    val model = Sequential()
    // input to an embedding layer is an index vector of `maxSeqquenceLength` elements, each index is in [0, vocabSize)
    // this layer produces a real-valued matrix of shape `maxSequenceLength x embeddingSize`
    model.add(Embedding(inputDim = vocabSize, outputDim = config.embeddingSize, inputLength=config.maxSequenceLength))
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

  /**
    * Reads an input text file and creates a data frame of two columns "x, y", where 
    * "x" are input token sequences and "y" are corresponding label sequences. The text file 
    * has a format of line-pair oriented: (y_i, x_i).
    *
    * @param sc a Spark context
    * @param config
    */
  def readData(sc: SparkContext, config: Config): DataFrame = {
    val spark = SparkSession.builder.getOrCreate()
    import spark.implicits._
    val df = sc.textFile(config.inputPath).zipWithIndex.toDF("line", "id")
    val df0 = df.filter(col("id") % 2 === 0).withColumn("y", col("line"))
    val df1 = df.filter(col("id") % 2 === 1).withColumn("x", col("line")).withColumn("id0", col("id") - 1)
    val af = df0.join(df1, df0.col("id") === df1.col("id0"))
    return af.select("x", "y")
  }

  def preprocess(df: DataFrame, config: Config): (PipelineModel, Array[String], Array[String]) = {
    val xTokenizer = new Tokenizer().setInputCol("x").setOutputCol("xs")
    val yTokenizer = new Tokenizer().setInputCol("y").setOutputCol("ys")
    val xVectorizer = new CountVectorizer().setInputCol("xs").setOutputCol("us").setMinDF(config.minFrequency)
      .setVocabSize(config.vocabSize).setBinary(true)
    val yVectorizer = new CountVectorizer().setInputCol("ys").setOutputCol("vs").setBinary(true)
    val pipeline = new Pipeline().setStages(Array(xTokenizer, yTokenizer, xVectorizer, yVectorizer))
    val model = pipeline.fit(df)
    val vocabulary = model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    val labels = model.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    println(s"vocabSize = ${vocabulary.size}, labels = ${labels.mkString}")
    return (model, vocabulary, labels)
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

        val df = VSC.readData(sc, config)
        df.printSchema()
        df.show()

        val (pipelineModel, vocabulary, labels) = VSC.preprocess(df, config)
        val vocabDict = vocabulary.zipWithIndex.toMap
        val af = pipelineModel.transform(df)
        val xSequencer = new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
        // map the label to one-based index (for use in ClassNLLCriterion)
        val labelDict = labels.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
        val ySequencer = new Sequencer(labelDict, config.maxSequenceLength, -1).setInputCol("ys").setOutputCol("label")
        val bf = ySequencer.transform(xSequencer.transform(af))
        bf.show()
        bf.printSchema()
        bf.select("label").show(5, false)

        // val Array(trainingDF, validationDF) = bf.randomSplit(Array(0.8, 0.2), seed = 80L)
      
        val model = tokenModel(vocabulary.size, labels.size, Shape(config.maxSequenceLength), config)

        val trainingSummary = TrainSummary(appName = getClass().getName(), logDir = "bin/vsc/sum/")
        val validationSummary = ValidationSummary(appName = getClass().getName(), logDir = "bin/vsc/sum/")
        val classifier = NNEstimator(model, TimeDistributedCriterion(ClassNLLCriterion(), true),
          Array(config.maxSequenceLength), Array(config.maxSequenceLength))
            .setLabelCol("label").setFeaturesCol("features")
            .setBatchSize(config.batchSize)
            .setOptimMethod(new Adam(config.learningRate))
            .setMaxEpoch(config.epochs)
            .setTrainSummary(trainingSummary)
            .setValidationSummary(validationSummary)
            .setValidation(Trigger.everyEpoch, bf, Array(new TimeDistributedTop1Accuracy(paddingValue = -1)), config.batchSize)
        classifier.fit(bf)

        sc.stop()
      case None => {}
    }
  }
}
