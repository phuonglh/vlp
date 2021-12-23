package vlp.nlm

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.dataset.{DataSet, SampleToMiniBatch, Sample}
import com.intel.analytics.bigdl.dataset.text.{Dictionary, TextToLabeledSentence, LabeledSentenceToSample}
import org.slf4j.LoggerFactory
import org.apache.log4j.Logger
import org.apache.log4j.Level
import vlp.tok.WordShape

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.{TimeDistributed, _}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger, Loss, Adam}

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import scopt.OptionParser


/**
  * phuonglh@gmail.com, December 2021
  * 
  * <p/>
  * A neural language model for Vietnamese.
  * 
  */
object LanguageModel {

    /**
      * Reads words from a line-separated text file, where each sentence is on a line and was 
      * word segmented using space. Tokens are normalized. An special padding token "<eos>" 
      * is appended to the end of each sentence.
      * @param fileName
      * @param syllableLevel
      * @return an iterator of words.
      */
    private def readWords(fileName: String, syllableLevel: Boolean = true): Iterator[String] = {
        val buffer = new ArrayBuffer[String]
        val lines = Source.fromFile(fileName).getLines
        val readWords = if (!syllableLevel) lines.foreach(x => {
            val words = x.split(" ").foreach(t => buffer.append(WordShape.normalize(t).toLowerCase()))
            buffer.append("<eos>")
        }) else lines.foreach(x => {
            val syllables = x.split(" ").foreach(t => {
              val word = WordShape.normalize(t).toLowerCase()
              val xs = word.split("_")
              xs.foreach(buffer.append(_))
            })
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
      * @param syllableLevel
      * @return an iterator of positive floats.
      */
    private def fileToWordIdx(fileName: String, dictionary: Dictionary, syllableLevel: Boolean = true): Iterator[Float] = {
        val words = readWords(fileName, syllableLevel)
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
        val words = readWords(options.trainDataPath, options.syllableLevel).toArray
        val dictionary = Dictionary(words, options.vocabSize - 1)
        dictionary.save(options.dictionaryPath + "/" + (if (options.syllableLevel) "syll" else "word"))
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

    private def transformer(options: OptionsLM): Module[Float] = {
        val input = Input[Float]()
        val transformer = Transformer[Float](vocabSize = options.vocabSize, hiddenSize = options.hiddenSize, numHeads = options.numHeads, 
              filterSize = options.hiddenSize*options.numHeads, numHiddenlayers = options.numLayers, 
              embeddingDropout = 1 - options.keepProb, attentionDropout = 0.1f, ffnDropout = 0.1f).inputs(input)
        val linear = Linear[Float](options.hiddenSize, options.vocabSize)
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
            transformer(options)
        } else {
            lstm(inputSize = options.vocabSize, hiddenSize = options.hiddenSize, outputSize = options.vocabSize, 
              numLayers = options.numLayers, keepProb = options.keepProb)
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
            val modelPath = options.checkpoint.get + "/" + (if (options.syllableLevel) "syll/" else "word/") + options.modelType
            optimizer.setCheckpoint(modelPath, Trigger.everyEpoch)
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

    def predict(seq: Seq[String], sc: SparkContext, model: Module[Float], dictionary: Dictionary, numSteps: Int): Tensor[Float] = {
      val words = seq.map(x => dictionary.getIndex(x).toFloat)
      val x = if (words.size > numSteps) words.toArray.slice(words.size - numSteps - 1, words.size) else {
        words.toArray ++ Array.fill[Float](numSteps - words.size + 1)(1.0f)
      }
      val sample = Sample(featureTensor = Tensor(x, Array(numSteps + 1)))
      val rdd = sc.parallelize(Seq(sample))
      val output = model.predict(rdd).map(_.toTensor[Float])
      output.first()
    }

    /**
      * Generates a number of syllables given first syllables. 
      *
      * @param firstSyllables
      * @param numSyllables
      * @param sc
      * @param model
      * @param dictionary
      * @param numSteps
      * @return a sequence of syllables
      */
    def generate(firstSyllables: Seq[String], numSyllables: Int, sc: SparkContext, model: Module[Float], dictionary: Dictionary, numSteps: Int): Seq[String] = {
      val buffer = new ArrayBuffer[Float]
      firstSyllables.foreach(s => buffer.append(dictionary.getIndex(s).toFloat))
      val id2Word = dictionary.index2Word()
      var stop = false
      while (!stop && buffer.size < firstSyllables.size + numSyllables) {
        val x = if (buffer.size > numSteps) buffer.toArray.slice(buffer.size - numSteps - 1, buffer.size) else {
          buffer.toArray ++ Array.fill[Float](numSteps - buffer.size + 1)(1.0f)
        }
        val sample = Sample(featureTensor = Tensor(x, Array(numSteps + 1)))
        val rdd = sc.parallelize(Seq(sample))
        val output = model.predict(rdd).map(_.toTensor[Float])
        val y = output.first()(buffer.size).toArray()
        val nextSyllable = y.zipWithIndex.maxBy(_._1)._2
        stop = nextSyllable == dictionary.getIndex("<eos>")
        buffer.append(nextSyllable.toFloat)
      }
      buffer.map(e => id2Word(e.toInt)).toSeq
    }



    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.WARN)
        val parser = new OptionParser[OptionsLM]("vlp.nlm.LanguageModel") {
            head("vlp.nlm.LanguageModel", "1.0")
            opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
            opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/predict")
            opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
            opt[String]('t', "modelType").action((x, conf) => conf.copy(modelType = x)).text("model type, either 'rnn' or 'trm' ")
            opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
            opt[Int]('u', "vocabSize").action((x, conf) => conf.copy(vocabSize = x)).text("number of words")
            opt[String]('d', "trainingDataPath").action((x, conf) => conf.copy(trainDataPath = x)).text("training data path")
            opt[String]('p', "checkpoint").action((x, conf) => conf.copy(checkpoint = Some(x))).text("checkpoint path")
            opt[Int]('h', "hiddenSize").action((x, conf) => conf.copy(hiddenSize = x)).text("hidden size")
            opt[Int]('n', "numSteps").action((x, conf) => conf.copy(numSteps = x)).text("maximum sequence length of a sentence")
            opt[Int]('k', "maxEpoch").action((x, conf) => conf.copy(maxEpoch = x)).text("number of epochs")
            opt[Unit]('v', "verbose").action((x, conf) => conf.copy(verbose = true)).text("verbose mode")
        }
        parser.parse(args, OptionsLM()) match {
            case Some(optionsLM) => 
                // create a BigDL-aware Spark context
                val conf = Engine.createSparkConf().setAppName(getClass().getName()).setMaster("local[*]")
                    .set("spark.rpc.message.maxSize", "200")
                val sc = new SparkContext(conf)
                Engine.init
                val spark = SparkSession.builder.config(conf).getOrCreate()
                optionsLM.mode match {
                  case "train" => train(sc, optionsLM)
                  case "eval" => 
                  case "predict" => 
                    // val seq = List("công_ty", "cung_cấp", "thiết_bị")
                    // val seq = List("công_ty", "nhận", "thấy", "hành_động", "này", "là")
                    val seq = List("đáng", "chú_ý", "trong", "các", "mặt_hàng", "bị")
                    // val seq = List("thủ_tướng", "yêu_cầu", "điều_tra")
                    // val seq = List("uỷ_ban", "kiểm_tra", "thành_uỷ", "cũng", "đã")
                    val subDir = (if (optionsLM.syllableLevel) "syll/" else "word/")
                    val modelPath = optionsLM.checkpoint.get + "/" + subDir + optionsLM.modelType + "/" + "20211221_163755/model.19901"
                    val model = Module.load[Float](modelPath) // NOTE: this is a deprecated method 
                    val dictionary = new Dictionary(optionsLM.dictionaryPath + "/" + subDir)
                    val tensor = predict(seq, sc, model, dictionary, optionsLM.numSteps)
                    val y = tensor(seq.size).toArray()
                    val id2Word = dictionary.index2Word()
                    val top5 = y.zipWithIndex.sortBy(_._1).reverse.take(5).map(p => id2Word(p._2) + " -> " + p._1)
                    println(top5.mkString(", "))
                  case "generate" => 
                    val seq = List("công_ty", "nhận", "thấy", "hành_động")
                    val subDir = (if (optionsLM.syllableLevel) "syll/" else "word/")
                    val modelPath = optionsLM.checkpoint.get + "/" + subDir + optionsLM.modelType + "/" + "20211221_163755/model.19901"
                    val model = Module.load[Float](modelPath) // NOTE: this is a deprecated method 
                    val dictionary = new Dictionary(optionsLM.dictionaryPath + "/" + subDir)
                    val output = generate(seq, 5, sc, model, dictionary, optionsLM.numSteps)
                    println(output)
                }
                spark.stop()
            case None => 
        }
    }
}
