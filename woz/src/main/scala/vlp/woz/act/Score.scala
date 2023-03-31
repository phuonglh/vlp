package vlp.woz.act

/**
  * Scores for multilabel classification.
  * phuonglh@gmail.com
  *
  */
case class Score(
  language: String,
  modelType: String,
  split: String,
  embeddingSize: Int = -1,    // applicable only for Token Model with RNN
  encoderSize: Int,           // recurrentSize in LSTM or hiddenSize in BERT
  layerSize: Int,             // number of recurrent layers in LSTM or number of blocks in BERT
  nHead: Int = -1,            // applicable for BERT
  intermediateSize: Int = -1, // applicable for BERT
  accuracy: Double,
  f1Measure: Double,
  microF1Measure: Double, 
  microPrecision: Double,
  microRecall: Double,
  precision: Array[Double],
  recall: Array[Double],
  fMeasure: Array[Double]
)