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
  def createModel(vocabSize: Int, labelSize: Int): Sequential[Float]
  def preprocessor(df: DataFrame): (PipelineModel, Array[String], Array[String])

  def predict(df: DataFrame, preprocessor: PipelineModel, bigdl: Sequential[Float]): DataFrame = {
    // add a custom layer ArgMax as the last layer of this model so as to 
    // make the nnframes API of BigDL work. By default, the BigDL nnframes only process 2-d data (including the batch dimension)
    bigdl.add(ArgMaxLayer())
    val vocabulary = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
    val vocabDict = vocabulary.zipWithIndex.toMap
    val bf = preprocessor.transform(df)
    val xSequencer = config.modelType match {
      case "tk" => 
        new Sequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
      case _ =>
        new MultiHotSequencer(vocabDict, config.maxSequenceLength, 0).setInputCol("xs").setOutputCol("features")
    }
    val cf = xSequencer.transform(bf)
    val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    val labelDict = labels.zipWithIndex.map(p => (p._1, p._2 + 1)).toMap
    // transform the gold "ys" labels to indices
    val ySequencer = new Sequencer(labelDict, config.maxSequenceLength, -1).setInputCol("ys").setOutputCol("label")    
    val ef = ySequencer.transform(cf)
    // run the prediction 
    val m = NNModel(bigdl)
    val ff = m.transform(ef)
    return ff.select("prediction", "label")
  }
}

object ModelFactory {
  def apply(representation: String, recurrent: Boolean, config: Config) = representation match {
    case "tk" => new TokenModel(config)
    case _ => new CharModel(config)
  }
  def apply(bigdl: KerasNet[Float], config: Config) = new TokenModel(config)
}

/**
  * Token-based model.
  *
  * @param config
  */
class TokenModel(config: Config) extends AbstractModel(config) {
  def createModel(vocabSize: Int, labelSize: Int): Sequential[Float] = {
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
    // model.add(TimeDistributed(
    //   Dense(config.hiddenSize, activation="relu").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
    //   inputShape=Shape(config.maxSequenceLength, config.recurrentSize))
    // )
    model.add(GRU(outputDim = config.hiddenSize, returnSequences = true))
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
  * Character-based model.
  *
  * @param config
  */
class CharModel(config: Config) extends AbstractModel(config) {
  def createModel(vocabSize: Int, labelSize: Int): Sequential[Float] = {
    val model = Sequential()
    // reshape the output to a matrix of shape `maxSequenceLength x 3*vocabSize`. This operation performs the concatenation 
    // of [b, i, e] embedding vectors (to [b :: i :: e]). Here vocab is the alphabet since each element is a character.
    model.add(Reshape(targetShape=Array(config.maxSequenceLength, 3*vocabSize), inputShape=Shape(3*config.maxSequenceLength*vocabSize)))
    // take the matrix above and feed to a GRU layer 
    // by default, the GRU layer produces a real-valued vector of length `recurrentSize` (the last output of the recurrent cell)
    // but since we want sequence information, we make it return a sequences, so the output will be a matrix of shape 
    // `maxSequenceLength x recurrentSize` 
    model.add(GRU(outputDim = config.recurrentSize, returnSequences = true))
    // // feed the output of the GRU to a dense layer with relu activation function
    // model.add(TimeDistributed(
    //   Dense(config.hiddenSize, activation="relu").asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], 
    //   inputShape=Shape(config.maxSequenceLength, config.recurrentSize))
    // )
    model.add(GRU(outputDim = config.hiddenSize, returnSequences = true))
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