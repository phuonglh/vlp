package vlp.vsc

import org.apache.spark.mllib.linalg.Matrix

case class Score(
  input: String,
  modelType: String,
  split: String,
  embeddingSize: Int = -1,    // applicable only for Token Model with RNN
  encoderSize: Int,           // recurrentSize in LSTM or hiddenSize in BERT
  layerSize: Int,             // number of recurrent layers in LSTM or number of blocks in BERT
  nHead: Int = -1,            // applicable for BERT
  intermediateSize: Int = -1, // applicable for BERT
  confusionMatrix: Matrix,
  accuracy: Double,
  precision: Array[Double],
  recall: Array[Double],
  fMeasure: Array[Double]
)