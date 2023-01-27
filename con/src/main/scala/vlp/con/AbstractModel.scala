package vlp.con

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
import org.apache.spark.ml.feature.{Tokenizer, RegexTokenizer, CountVectorizer, CountVectorizerModel, StringIndexer}

/**
 * Spelling error detection model.
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
    val xSequencer = config.modelType match {
      case "tk" => new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")        
      case "tb" => new Sequencer4BERT(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")        
      case _ => new MultiHotSequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")        
    }
    val cf = xSequencer.transform(bf)
    val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    val labelDict = labels.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    // transform the gold "ys" labels to indices. NOTE: this is for evaluation mode only
    val ySequencer = new Sequencer(labelDict, config.maxSequenceLength, -1).setInputCol("ys").setOutputCol("label")    
    val ef = ySequencer.transform(cf)

    // add a custom layer ArgMax as the last layer of the BigDL model so as to 
    // make the nnframes API of BigDL work. By default, the BigDL nnframes only process 2-d data (including the batch dimension)
    val m = if (config.modelType == "tk" || config.modelType == "ch") {
      val sequential = bigdl.asInstanceOf[Sequential[Float]]
      sequential.add(ArgMaxLayer())
      println(sequential.summary())
      NNModel(sequential)
    } else {
      val model = bigdl.asInstanceOf[Model[Float]]
      val inputs = model.nodes(Seq("inputIds", "segmentIds", "positionIds", "masks"))
      val output = model.node("output")
      val outputNew = ArgMaxLayer().setName("argMax").inputs(output)
      val modelNew = Model(inputs.toArray, outputNew)
      println(modelNew.summary())
      // we need to provide feature size for this multiple-output module (to convert 'features' into a table)
      val maxSeqLen = config.maxSequenceLength
      val featureSize = Array(Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen), Array(maxSeqLen))
      NNModel(modelNew, featureSize)
    }
    val ff = m.transform(ef)
    return ff.select("prediction", "label")
  }
}

object ModelFactory {
  def apply(representation: String, config: Config) = representation match {
    case "tk" => new TokenModel(config)
    case "ch" => new CharModel(config)
    case "tb" => new TokenModelBERT(config)
    case _ => new TokenModel(config)
  }
  def apply(bigdl: KerasNet[Float], config: Config) = new TokenModel(config)
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
    model.add(Embedding(inputDim = vocabSize, outputDim = config.embeddingSize, inputLength=config.maxSequenceLength))
    // take the matrix above and feed to a RNN layer 
    // by default, the RNN layer produces a real-valued vector of length `recurrentSize` (the last output of the recurrent cell)
    // but since we want sequence information, we make it return a sequences, so the output will be a matrix of shape 
    // `maxSequenceLength x recurrentSize` 
    model.add(LSTM(outputDim = config.recurrentSize, returnSequences = true))
    // feed the output of the RNN to a dense layer with relu activation function
    // model.add(TimeDistributed(
    //   Dense(config.hiddenSize, activation="relu").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
    //   inputShape=Shape(config.maxSequenceLength, config.recurrentSize))
    // )
    model.add(LSTM(outputDim = config.hiddenSize, returnSequences = true))
    // add a dropout layer for regularization
    model.add(Dropout(config.dropoutProbability))
    // add the last layer for multi-class classification
    model.add(TimeDistributed(
      Dense(labelSize, activation="softmax").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
      inputShape=Shape(config.maxSequenceLength, config.hiddenSize))
    )
    return model
  }

  def preprocessor(df: DataFrame): (PipelineModel, Array[String], Array[String]) = {
    val xTokenizer = new Tokenizer().setInputCol("x").setOutputCol("xs")
    val xVectorizer = new CountVectorizer().setInputCol("xs").setOutputCol("us").setMinDF(config.minFrequency)
      .setVocabSize(config.vocabSize).setBinary(true)
    val yTokenizer = new Tokenizer().setInputCol("y").setOutputCol("ys")
    val yVectorizer = new CountVectorizer().setInputCol("ys").setOutputCol("vs").setBinary(true)
    val pipeline = new Pipeline().setStages(Array(xTokenizer, xVectorizer, yTokenizer, yVectorizer))
    val preprocessor = pipeline.fit(df)
    val vocabulary = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
    val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    println(s"vocabSize = ${vocabulary.size}, labels = ${labels.mkString}")
    return (preprocessor, vocabulary, labels)
  }
}

/**
  * Character-based model using LSTM. This is a sequential model.
  *
  * @param config
  */
class CharModel(config: Config) extends AbstractModel(config) {
  def createModel(vocabSize: Int, labelSize: Int): KerasNet[Float] = {
    val model = Sequential()
    // reshape the output to a matrix of shape `maxSequenceLength x 3*vocabSize`. This operation performs the concatenation 
    // of [b, i, e] embedding vectors (to [b :: i :: e]). Here vocab is the alphabet since each element is a character.
    model.add(Reshape(targetShape=Array(config.maxSequenceLength, 3*vocabSize), inputShape=Shape(3*config.maxSequenceLength*vocabSize)))
    // take the matrix above and feed to a RNN layer 
    // by default, the RNN layer produces a real-valued vector of length `recurrentSize` (the last output of the recurrent cell)
    // but since we want sequence information, we make it return a sequences, so the output will be a matrix of shape 
    // `maxSequenceLength x recurrentSize` 
    model.add(LSTM(outputDim = config.recurrentSize, returnSequences = true))
    // // feed the output of the RNN to a dense layer with relu activation function
    // model.add(TimeDistributed(
    //   Dense(config.hiddenSize, activation="relu").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
    //   inputShape=Shape(config.maxSequenceLength, config.recurrentSize))
    // )
    model.add(LSTM(outputDim = config.hiddenSize, returnSequences = true))
    // add a dropout layer for regularization
    model.add(Dropout(config.dropoutProbability))
    // add the last layer for multi-class classification
    model.add(TimeDistributed(
      Dense(labelSize, activation="softmax").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
      inputShape=Shape(config.maxSequenceLength, config.hiddenSize))
    )
    return model
  }

  def preprocessor(df: DataFrame): (PipelineModel, Array[String], Array[String]) = {
    val xTokenizer = new RegexTokenizer().setInputCol("x").setOutputCol("cs").setPattern(".").setGaps(false).setToLowercase(true)
    val xVectorizer = new CountVectorizer().setInputCol("cs").setOutputCol("us").setMinDF(config.minFrequency)
      .setVocabSize(config.vocabSize).setBinary(true)
    val yTokenizer = new Tokenizer().setInputCol("y").setOutputCol("ys")
    val yVectorizer = new CountVectorizer().setInputCol("ys").setOutputCol("vs").setBinary(true)
    val tokenizer = new Tokenizer().setInputCol("x").setOutputCol("xs")
    val pipeline = new Pipeline().setStages(Array(xTokenizer, xVectorizer, yTokenizer, yVectorizer, tokenizer))
    val preprocessor = pipeline.fit(df)
    val vocabulary = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
    val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    println(s"vocabSize = ${vocabulary.size}, labels = ${labels.mkString}")
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

    // reshape the vector of length 4*maxSeqLen to a matrix of shape Array(4, maxSeqLen)
    // val reshape = Reshape(targetShape=Array(4, config.maxSequenceLength), inputShape=Shape(4*config.maxSequenceLength))    
    // model.add(reshape)
    // // split the matrix to a table of 4 inputs. We split along dimension 0 (row)
    // val split = SplitTensor(0, 4)
    // model.add(split)
    // // delete the singleton dimension 1 (the batch dimension is 0)
    // model.add(SqueezeTableLayer())
    // feed the table to a BERT layer, output the last block state only
    val hiddenSize = config.bert.hiddenSize
    val nBlock = config.bert.nBlock
    val nHead = config.bert.nHead
    val maxPositionLen = config.bert.maxPositionLen
    val intermediateSize = config.bert.intermediateSize
    val bert = BERT(vocabSize, hiddenSize, nBlock, nHead, maxPositionLen, intermediateSize, outputAllBlock = false).setName("BERT")
    val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masksReshaped))
    val bertOutput = SelectTable(0).setName("firstBlock").inputs(bertNode)

    val dense = Dense(labelSize).setName("dense").inputs(bertOutput)
    val output = SoftMax().setName("output").inputs(dense)
    val model = Model(Array(inputIds, segmentIds, positionIds, masks), output)
    return model
  }
}
