package vlp.woz.act

/**
  * Scoes for multilabel classification.
  * phuonglh@gmail.com
  *
  * @param modelType
  * @param split
  * @param embeddingSize
  * @param encoderSize
  * @param layerSize
  * @param nHead
  * @param intermediateSize
  * @param accuracy
  * @param precision
  * @param recall
  * @param fMeasure
  */
case class Score(
  modelType: String,
  split: String,
  embeddingSize: Int = -1,    // applicable only for Token Model with RNN
  encoderSize: Int,           // recurrentSize in LSTM or hiddenSize in BERT
  layerSize: Int,             // number of recurrent layers in LSTM or number of blocks in BERT
  nHead: Int = -1,            // applicable for BERT
  intermediateSize: Int = -1, // applicable for BERT
  accuracy: Double,
  precision: Array[Double],
  recall: Array[Double],
  fMeasure: Array[Double]
)