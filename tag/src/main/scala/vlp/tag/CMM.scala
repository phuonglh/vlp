package vlp.tag

import org.apache.spark.ml.linalg.{Matrix, Vector}
import scala.collection.mutable.ListBuffer

/**
  * Conditional Markov model which predicts tag sequence for a given word sequence.
  *
  * phuonglh
  */
class CMM(val vocabulary: Map[String, Int],
          val weights: Matrix,
          val intercept: Vector,
          val labels: Seq[String],
          val featureTypes: Seq[String],
          val markovOrder: Int,
          val verbose: Boolean = false) extends Serializable {

  /**
    * Greedy decoding approach.
    * @param ws
    * @param ts
    * @return best tag sequence.
    */
  def predict(ws: Seq[String], ts: Array[String]): Seq[String] = {
    val words = new ListBuffer[String]
    words ++= ws

    for (j <- 0 until words.size) {
      val features = FeatureExtractor.extract(words, ts, featureTypes, markovOrder, j)
      val prediction = classify(features)
      ts(j) = prediction
    }
    ts
  }

  def classify(features: Seq[String]): String = {
    val fs = features.map(_.toLowerCase).map(f => vocabulary.getOrElse(f, -1)).filter(_ > 0)
    val numLabels = intercept.size
    val scores = (0 until numLabels).map {
      k => fs.map(j => weights(k, j)).sum + intercept(k)
    }
    val bestLabel = scores.zipWithIndex.maxBy(_._1)._2
    labels(bestLabel)
  }
}
