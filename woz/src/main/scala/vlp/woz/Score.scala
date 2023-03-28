package vlp.woz

/**
  * Scores for multilabel classification.
  * 
  * phuonglh@gmail.com
  *
  */
case class Score(
  language: String,
  modelType: String,
  split: String,
  accuracy: Double,
  f1Measure: Double,
  microF1Measure: Double, 
  microPrecision: Double,
  microRecall: Double,
  precision: Array[Double],
  recall: Array[Double],
  fMeasure: Array[Double]
)