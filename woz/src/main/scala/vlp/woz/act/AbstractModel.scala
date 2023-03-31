package vlp.woz.act

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.{Sequential, Model}
import com.intel.analytics.bigdl.dllib.keras.models.KerasNet
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.nnframes.NNModel

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{Tokenizer, RegexTokenizer, CountVectorizer, CountVectorizerModel, StringIndexer, StringIndexerModel, VectorAssembler}

import vlp.woz.WordShaper


/**
 * Multi-label dialog act classification.
 * 
  * phuonglh@gmail.com
  */
abstract class AbstractModel(config: Config) {
  def createModel(vocabSize: Int, labelSize: Int): KerasNet[Float]
  def preprocessor(df: DataFrame): (PipelineModel, Array[String], Array[String])

  def predict(df: DataFrame, preprocessor: PipelineModel, bigdl: KerasNet[Float]): DataFrame = {
    val vocabulary = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    val vocabDict = vocabulary.zipWithIndex.toMap
    val bf = preprocessor.transform(df)
    // transform the input data frame into features
    val cf = config.modelType match {
      case "lstm" => 
        val xSequencer = new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
        xSequencer.transform(bf)
      case "bert" => 
        val xSequencer = new Sequencer4BERT(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
        xSequencer.transform(bf)
      case "lstm-boa" =>
        val xSequencer = new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("ts")
        val bf1 = xSequencer.transform(bf)
        val assembler = new VectorAssembler().setInputCols(Array("ts", "ys")).setOutputCol("features")
        assembler.transform(bf1)
      case _ => 
        // default to TokenModel (lstm)
        val xSequencer = new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
        xSequencer.transform(bf)
    }
    cf.show()
    val m = if (config.modelType == "lstm") {
      println(bigdl.summary())
      NNModel(bigdl)
    } else if (config.modelType == "bert") {
      // we need to provide feature size for this multiple-input module (to convert 'features' into a table)
      val maxSeqLen = config.maxSequenceLength
      val featureSize = Array(Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen))
      println(bigdl.summary())
      NNModel(bigdl, featureSize)
    } else { // lstm-boa
      val prevActs = preprocessor.stages(4).asInstanceOf[CountVectorizerModel].vocabulary
      val featureSize = Array(Array(config.maxSequenceLength), Array(prevActs.size))
      println(bigdl.summary())
      NNModel(bigdl, featureSize)
    }
    // run the prediction and return predicted labels as well as gold labels
    val ff = m.transform(cf)
    // val selector = new TopKSelector(2).setInputCol("prediction").setOutputCol("output")
    val selector = new ThresholdSelector().setInputCol("prediction").setOutputCol("output")
    val gf = selector.transform(ff)    
    return gf.select("output", "target")
  }
}

object ModelFactory {
  /**
    * Create a model from scratch.
    *
    * @param config
    * @return a model
    */
  def apply(config: Config) = config.modelType match {
    case "lstm" => new TokenModel(config)
    case "bert" => new TokenModelBERT(config)
    case "lstm-boa" => new TokenModelBOA(config)
    case _ => new TokenModel(config)
  }
}

/**
  * Token-based model using LSTM. This is a sequential model.
  *
  * @param config
  */
class TokenModel(config: Config) extends AbstractModel(config) {
  def createModel(vocabSize: Int, labelSize: Int): KerasNet[Float] = {
    val model = Sequential()
    // input to an embedding layer is an index vector of `maxSequenceLength` elements, each index is in [0, vocabSize)
    // this layer produces a real-valued matrix of shape `maxSequenceLength x embeddingSize`
    model.add(Embedding(inputDim = vocabSize, outputDim = config.embeddingSize, inputLength=config.maxSequenceLength).setName(s"Embedding-${config.modelType}"))
    // take the embedding matrix above and feed to a RNN layer 
    // by default, the RNN layer produces a real-valued vector of length `recurrentSize` (the last output of the recurrent cell)
    // if using multi-layers of LSTM, we need to use returnSequences=true except for the last layer.
    for (j <- 0 until (config.layers-1))
      model.add(LSTM(outputDim = config.recurrentSize, returnSequences = true).setName(s"LSTM-$j"))
    model.add(LSTM(outputDim = config.recurrentSize).setName(s"LSTM-${config.layers-1}"))
    // add a dropout layer for regularization
    model.add(Dropout(config.dropoutProbability).setName("Dropout"))
    // add the last layer for BCE loss
    // sigmoid for multi-label instead of softmax, which gives better performance
    model.add(Dense(labelSize, activation="sigmoid").setName("Dense"))  
    return model
  }

  def preprocessor(df: DataFrame): (PipelineModel, Array[String], Array[String]) = {
    val xTokenizer = new RegexTokenizer().setInputCol("utterance").setOutputCol("tokens").setPattern("""[\s,?.'"/!;)(]+""")
    val xShaper = new WordShaper().setInputCol("tokens").setOutputCol("xs")
    val xVectorizer = new CountVectorizer().setInputCol("xs").setOutputCol("us").setMinDF(config.minFrequency).setVocabSize(config.vocabSize).setBinary(true)
    val yVectorizer = new CountVectorizer().setInputCol("actNames").setOutputCol("label").setBinary(true)
    val pipeline = new Pipeline().setStages(Array(xTokenizer, xShaper, xVectorizer, yVectorizer))
    val preprocessor = pipeline.fit(df)
    val vocabulary = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    println(s"vocabSize = ${vocabulary.size}, labels = ${labels.mkString(", ")}")
    return (preprocessor, vocabulary, labels)
  }
}

/**
  * The token-based BERT model extends [[TokenModel]] to reuse its preprocessor. This is a graph model.
  *
  * @param config
  */
class TokenModelBERT(config: Config) extends TokenModel(config) {
  override def createModel(vocabSize: Int, labelSize: Int): KerasNet[Float] = {
    val maxSeqLen = config.maxSequenceLength
    val inputIds = Input(inputShape = Shape(maxSeqLen), "inputIds")
    val segmentIds = Input(inputShape = Shape(maxSeqLen), "segmentIds")
    val positionIds = Input(inputShape = Shape(maxSeqLen), "positionIds")
    val masks = Input(inputShape = Shape(maxSeqLen), "masks")
    val masksReshaped = Reshape(targetShape = Array(1, 1, maxSeqLen)).setName("reshape").inputs(masks)

    val hiddenSize = config.bert.hiddenSize
    val nBlock = config.bert.nBlock
    val nHead = config.bert.nHead
    val maxPositionLen = config.bert.maxPositionLen
    val intermediateSize = config.bert.intermediateSize
    // use a BERT layer, not output all blocks (there will be 2 outputs)
    val bert = BERT(vocabSize, hiddenSize, nBlock, nHead, maxPositionLen, intermediateSize, outputAllBlock = false).setName("BERT")
    val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masksReshaped))
    // get the pooled output which processes the hidden state of the last layer with regard to the first
    //  token of the sequence. This would be useful for classification tasks.
    val bertOutput = SelectTable(1).setName("firstBlock").inputs(bertNode)
    val output = Dense(labelSize, activation="sigmoid").setName("output").inputs(bertOutput)
    val model = Model(Array(inputIds, segmentIds, positionIds, masks), output)
    return model
  }
}

/**
  * Extended token model which uses LSTM: Bag-of-Act representation.
  *
  * @param config
  */
class TokenModelBOA(config: Config) extends AbstractModel(config) {

  override def createModel(vocabSize: Int, labelSize: Int): KerasNet[Float] = {
    val maxSeqLen = config.maxSequenceLength
    // utterance token ids
    val xs = Input(inputShape = Shape(maxSeqLen), "xs")
    // previous act ids
    val ps = Input(inputShape = Shape(labelSize), "ps")
    // embed xs to es vectors
    // input to an embedding layer is an index vector of `maxSequenceLength` elements, each index is in [0, vocabSize)
    // this layer produces a real-valued matrix of shape `maxSequenceLength x embeddingSize`
    val es = Embedding(inputDim = vocabSize, outputDim = config.embeddingSize, inputLength=maxSeqLen).setName(s"Embedding-${config.modelType}").inputs(xs)
    // take the embedding matrix and feed to a RNN layer 
    // by default, the RNN layer produces a real-valued vector of length `recurrentSize` (the last output of the recurrent cell)
    // if using multi-layers of LSTM, we need to use returnSequences=true except for the last layer.
    // first recurrent layer
    val r0 = LSTM(outputDim = config.recurrentSize, returnSequences = true).setName("LSTM-0").inputs(es)
    // second recurrent layer
    val r1 = LSTM(outputDim = config.recurrentSize).setName("LSTM-1").inputs(r0)
    // concat the default last dimension
    val concat = Merge(mode="concat").setName("Concat").inputs(r1, ps)
    // output
    val output = Dense(labelSize, activation="sigmoid").setName("Output").inputs(concat)
    val model = Model(Array(xs, ps), output)
    return model
  }

  override def preprocessor(df: DataFrame): (PipelineModel, Array[String], Array[String]) = {
    val xTokenizer = new RegexTokenizer().setInputCol("utterance").setOutputCol("tokens").setPattern("""[\s,?.'"/!;)(]+""")
    val xShaper = new WordShaper().setInputCol("tokens").setOutputCol("xs")
    val xVectorizer = new CountVectorizer().setInputCol("xs").setOutputCol("us").setMinDF(config.minFrequency).setVocabSize(config.vocabSize).setBinary(true)
    val yVectorizer = new CountVectorizer().setInputCol("actNames").setOutputCol("label").setBinary(true)
    // prevAct vectorizer    
    val prevVectorizer = new CountVectorizer().setInputCol("ps").setOutputCol("ys").setBinary(true)
    val pipeline = new Pipeline().setStages(Array(xTokenizer, xShaper, xVectorizer, yVectorizer, prevVectorizer))
    val preprocessor = pipeline.fit(df)
    val vocabulary = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    println(s"vocabSize = ${vocabulary.size}, labels = ${labels.mkString(", ")}")
    return (preprocessor, vocabulary, labels)
  }
}
