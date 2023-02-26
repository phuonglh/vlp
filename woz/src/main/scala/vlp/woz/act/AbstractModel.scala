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


import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{Tokenizer, RegexTokenizer, CountVectorizer, CountVectorizerModel, StringIndexer, StringIndexerModel}

/**
 * Multi-label dialog act classification.
 * 
  * phuonglh@gmail.com
  */
abstract class AbstractModel(config: Config) {
  def createModel(vocabSize: Int, labelSize: Int): KerasNet[Float]
  def preprocessor(df: DataFrame): (PipelineModel, Array[String], Array[String])

  def predict(df: DataFrame, preprocessor: PipelineModel, bigdl: KerasNet[Float]): DataFrame = {
    val vocabulary = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
    val vocabDict = vocabulary.zipWithIndex.toMap
    val bf = preprocessor.transform(df)
    // use a sequencer to transform the input data frame into features
    val xSequencer = if (config.modelType == "lstm") {
      new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
    } else {
      new Sequencer4BERT(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
    }
    val cf = xSequencer.transform(bf)
    cf.show()
    val m = if (config.modelType == "lstm") {
      println(bigdl.summary())
      NNModel(bigdl)
    } else {
      // val model = bigdl.asInstanceOf[Model[Float]]
      // we need to provide feature size for this multiple-input module (to convert 'features' into a table)
      val maxSeqLen = config.maxSequenceLength
      val featureSize = Array(Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen))
      println(bigdl.summary())
      NNModel(bigdl, featureSize)
    }
    // run the prediction and return predicted labels as well as gold labels
    val ff = m.transform(cf)
    val selector = new TopKSelector(2).setInputCol("prediction").setOutputCol("output")
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
    // input to an embedding layer is an index vector of `maxSeqquenceLength` elements, each index is in [0, vocabSize)
    // this layer produces a real-valued matrix of shape `maxSequenceLength x embeddingSize`
    model.add(Embedding(inputDim = vocabSize, outputDim = config.embeddingSize, inputLength=config.maxSequenceLength).setName(s"Embedding-${config.modelType}"))
    // take the matrix above and feed to a RNN layer 
    // by default, the RNN layer produces a real-valued vector of length `recurrentSize` (the last output of the recurrent cell)
    for (j <- 0 until config.layers)
      model.add(LSTM(outputDim = config.recurrentSize).setName(s"LSTM-$j"))
    // add a dropout layer for regularization
    model.add(Dropout(config.dropoutProbability).setName("dropout"))
    // add the last layer for multi-class classification
    model.add(Dense(labelSize, activation="softmax").setName("Dense"))    
    return model
  }

  def preprocessor(df: DataFrame): (PipelineModel, Array[String], Array[String]) = {
    val xTokenizer = new RegexTokenizer().setInputCol("utterance").setOutputCol("xs").setPattern("""[\s,?.'"/!;)(]+""")
    val xVectorizer = new CountVectorizer().setInputCol("xs").setOutputCol("us").setMinDF(config.minFrequency).setVocabSize(config.vocabSize).setBinary(true)
    val yVectorizer = new CountVectorizer().setInputCol("actNames").setOutputCol("label").setBinary(true)
    val pipeline = new Pipeline().setStages(Array(xTokenizer, xVectorizer, yVectorizer))
    val preprocessor = pipeline.fit(df)
    val vocabulary = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
    val labels = preprocessor.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
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
    val dense = Dense(labelSize).setName("dense").inputs(bertOutput)
    val output = SoftMax().setName("output").inputs(dense)
    val model = Model(Array(inputIds, segmentIds, positionIds, masks), output)
    return model
  }
}
