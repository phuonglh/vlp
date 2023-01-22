package vlp.con

import org.apache.spark.mllib.linalg.Matrix

case class Score(
  confusionMatrix: Matrix,
  accuracy: Double,
  precisions: Array[Double],
  recalls: Array[Double],
  fMeasures: Array[Double]
)