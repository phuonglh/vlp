package vlp.med

/**
 * Scores for multilabel classification.
 *
 * phuonglh@gmail.com
 *
 */
case class Score(
  language: String,
  split: String,
  accuracy: Double,
  weightedFMeasure: Double,
  precision: Array[Double],
  recall: Array[Double],
  fMeasure: Array[Double]
)
