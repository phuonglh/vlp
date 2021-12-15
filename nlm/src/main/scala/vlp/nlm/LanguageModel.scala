package vlp.nlm

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dataset.{DataSet, SampleToMiniBatch}
import com.intel.analytics.bigdl.dataset.text.{Dictionary, TextToLabeledSentence, LabeledSentenceToSample}
import com.intel.analytics.bigdl.dataset.SampleToMiniBatch
import org.slf4j.LoggerFactory
import org.apache.log4j.Logger
import org.apache.log4j.Level

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.{TimeDistributed, _}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger, Loss, Adam}

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession


case class OptionsLM(
    trainDataPath: String = "dat/txt/vlsp.jul.tok",
    validDataPath: String = "dat/txt/vlsp.jul.tok",
    dictionaryPath: String = "dat/txt",
    vocabSize: Int = 10000,
    numSteps: Int = 20,
    batchSize: Int = 128,
    maxEpoch: Int = 40,
    modelType: String = "rnn", // ["rnn", "trm"]
    numLayers: Int = 2,
    hiddenSize: Int = 256,
    keepProb: Float = 2.0f,
    checkpoint: Option[String] = Some("models/nlm"),
    overWriteCheckpoint: Boolean = true,
    learningRate: Double = 1E-3,
)

/**
  * phuonglh@gmail.com, December 2021
  * 
  */
object LanguageModel {

    /**
      * Reads words from a line-separated text file, where each sentence is on a line and was 
      * word segmented using space. An special padding token "<eos>" is appended to the end of 
      * each sentence.
      * @param fileName
      * @return an iterator of words.
      */
    private def readWords(fileName: String): Iterator[String] = {
        val buffer = new ArrayBuffer[String]
        val readWords = Source.fromFile(fileName).getLines.foreach(x => {
            val words = x.split(" ").foreach(t => buffer.append(t.toLowerCase()))
            buffer.append("<eos>")
        })
        buffer.toIterator
    }

    /**
      * Reads and converts words in a text file into float numbers using a dictionary.
      * The word indices start from 1.0f. 
      *
      * @param fileName
      * @param dictionary
      * @return an iterator of positive floats.
      */
    private def fileToWordIdx(fileName: String, dictionary: Dictionary): Iterator[Float] = {
        val words = readWords(fileName)
        words.map(x => dictionary.getIndex(x).toFloat + 1.0f)
    }

    /**
      * Splits a raw sentence into chunks of <code>numSteps</code>.
      *
      * @param rawData
      * @param numSteps
      * @return an array of numSteps-size arrays.
      */
    private def split(rawData: Array[Float], numSteps: Int): Array[Array[Float]] = {
        var offset = 0
        val length = rawData.length - 1 - numSteps
        val buffer = new ArrayBuffer[Array[Float]]
        while (offset <= length) {
            val slice = new Array[Float](numSteps + 1)
            Array.copy(rawData, offset, slice, 0, numSteps + 1)
            buffer.append(slice)
            offset += numSteps
        }
        buffer.toArray[Array[Float]]
    }

    def createDatasets(sc: SparkContext, options: OptionsLM) = {
        val words = readWords(options.trainDataPath).toArray
        val dictionary = Dictionary(words, options.vocabSize - 1)
        dictionary.save(options.dictionaryPath)
        val trainData = fileToWordIdx(options.trainDataPath, dictionary).toArray
        val validData = fileToWordIdx(options.validDataPath, dictionary).toArray

        val trainSet = DataSet.rdd(sc.parallelize(split(trainData, options.numSteps)))
            .transform(TextToLabeledSentence[Float](options.numSteps))
            .transform(LabeledSentenceToSample[Float](oneHot = false, fixDataLength = None, fixLabelLength = None))
            .transform(SampleToMiniBatch[Float](options.batchSize))
        val validSet = DataSet.rdd(sc.parallelize(split(validData, options.numSteps)))
            .transform(TextToLabeledSentence[Float](options.numSteps))
            .transform(LabeledSentenceToSample[Float](oneHot = false, fixDataLength = None, fixLabelLength = None))
            .transform(SampleToMiniBatch[Float](options.batchSize))
        (trainSet, validSet)
    }

    private def transformer(inputSize: Int = 10000, hiddenSize: Int = 256, outputSize: Int = 10000, numLayers: Int = 2, keepProb: Float = 2.0f): Module[Float] = {
        val input = Input[Float]()
        val transformer = Transformer[Float](vocabSize = inputSize, hiddenSize = hiddenSize, numHeads = 4, filterSize = hiddenSize*4,
                numHiddenlayers = numLayers, embeddingDropout = 1 - keepProb, attentionDropout = 0.1f, ffnDropout = 0.1f)
            .inputs(input)
        val linear = Linear[Float](hiddenSize, outputSize)
        val output = TimeDistributed[Float](linear).inputs(transformer)
        Graph(input, output)
    }

    private def lstm(inputSize: Int, hiddenSize: Int, outputSize: Int, numLayers: Int, keepProb: Float = 2.0f): Module[Float] = {
        val input = Input[Float]()
        val embeddingLookup = LookupTable[Float](inputSize, hiddenSize).inputs(input)

        val inputs = if (keepProb < 1) {
            Dropout[Float](keepProb).inputs(embeddingLookup)
        } else embeddingLookup

        val lstm = addLayer(hiddenSize, hiddenSize, 1, numLayers, inputs)
        val linear = Linear[Float](hiddenSize, outputSize)
        val output = TimeDistributed[Float](linear).inputs(lstm)

        val model = Graph(input, output)
        model.asInstanceOf[StaticGraph[Float]].setInputFormats(Seq(Memory.Format.nc))
        model.asInstanceOf[StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.ntc))
        model
    }

    /**
      * Adds and chains layers recursively.
      *
      * @param inputSize
      * @param hiddenSize
      * @param depth
      * @param numLayers
      * @param input
      * @return a module
      */
    private def addLayer(inputSize: Int, hiddenSize: Int, depth: Int, numLayers: Int, input: ModuleNode[Float]): ModuleNode[Float] = {
        if (depth == numLayers) {
            Recurrent[Float]().add(LSTM[Float](inputSize, hiddenSize, 0, null, null, null)).inputs(input)
        } else {
            addLayer(inputSize, hiddenSize, depth + 1, numLayers, Recurrent[Float]()
                .add(LSTM[Float](inputSize, hiddenSize, 0, null, null, null)).inputs(input)
            )
        }
    }

    def createModel(options: OptionsLM): Module[Float] = {
        if (options.modelType == "trm") {
            transformer(inputSize = options.vocabSize, hiddenSize = options.hiddenSize, outputSize = options.vocabSize, numLayers = options.numLayers, keepProb = options.keepProb)
        } else {
            lstm(inputSize = options.vocabSize, hiddenSize = options.hiddenSize, outputSize = options.vocabSize, numLayers = options.numLayers, keepProb = options.keepProb)
        }
    }

    /**
      * Trains a language model.
      *
      * @param sc
      * @param options
      * @return a module.
      */
    def train(sc: SparkContext, options: OptionsLM): Module[Float] = {
        val (trainSet, validationSet) = createDatasets(sc, options)
        val optimizer = Optimizer(model = createModel(options), dataset = trainSet,
            criterion = TimeDistributedCriterion[Float](CrossEntropyCriterion[Float](), sizeAverage = false, dimension = 1)
        )
        if (options.checkpoint.isDefined) {
            optimizer.setCheckpoint(options.checkpoint.get, Trigger.everyEpoch)
        }
        if (options.overWriteCheckpoint) {
            optimizer.overWriteCheckpoint()
        }
        optimizer.setValidation(Trigger.everyEpoch, validationSet, 
            Array(new Loss[Float](TimeDistributedCriterion[Float](CrossEntropyCriterion[Float](), sizeAverage = false, dimension = 1)))
        ).setOptimMethod(new Adam(learningRate = options.learningRate))
            .setEndWhen(Trigger.maxEpoch(options.maxEpoch))
            .optimize()
    }

    def predict(seq: Seq[String], sc: SparkContext, options: OptionsLM) = {

    }

    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.WARN)
        // create a BigDL-aware Spark context
        val conf = Engine.createSparkConf().setAppName(getClass().getName()).setMaster("local[*]")
            .set("spark.rpc.message.maxSize", "200")
        val sc = new SparkContext(conf)
        Engine.init
        val spark = SparkSession.builder.config(conf).getOrCreate()
        // train a model
        train(sc, OptionsLM())

        spark.stop()
    }
}
