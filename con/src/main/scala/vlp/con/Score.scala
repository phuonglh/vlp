package vlp.con

import org.apache.spark.mllib.linalg.Matrix

case class Score(
  input: String,
  modelType: String,
  split: String,
  embeddingSize: Int = -1,  // applicable only for Token Model with RNN
  encoderSize: Int,         // recurrentSize in LSTM or hiddenSize in BERT
  layerSize: Int,           // number of recurrent layers in LSTM or number of blocks in BERT
  confusionMatrix: Matrix,
  accuracy: Double,
  precision: Array[Double],
  recall: Array[Double],
  fMeasure: Array[Double]
)