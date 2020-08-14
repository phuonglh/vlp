package vlp.ner

import scala.collection.mutable.ListBuffer

/**
  * phuonglh, 3/19/18, 23:31
  * 
  * A beam search implementation.
  */
object BeamSearch {
  /**
    * Beam search decoding.
    * @param scores a double lattice of shape mxn, where m is the number of labels
    *              and n is the number of steps. The scores are in log scale so that 
    *              we can add values instead of multiply them.
    * @param k beam size
    * @return a list of k elements, each element is a tuple of path and its corresponding score.
    */
  def decode(scores: Array[Array[Double]], k: Int = 5): List[(List[Int], Double)] = {
    var paths = new ListBuffer[(List[Int], Double)]
    paths.append((List.empty[Int], 0))
    val m = scores.length
    val n = scores(0).length
    for (j <- 0 until n) {
      val candidates = new ListBuffer[(List[Int], Double)]
      for (u <- 0 until paths.length) {
        val (seq, score) = paths(u)
        for (i <- 0 until m) {
          val candidate = (seq :+ i, score + scores(i)(j))
          candidates.append(candidate)
        }
      }
      val orderedCandidates = candidates.sortBy(_._2).reverse
      paths = orderedCandidates.take(k)
    }
    paths.toList
  }

  def main(args: Array[String]): Unit = {
    val scores = Array.ofDim[Double](5, 10)
    scores(0) = Array(0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5)
    scores(1) = Array(0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4)
    scores(2) = Array(0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3)
    scores(3) = Array(0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2)
    scores(4) = Array(0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1)
    val result = decode(scores)
    result.foreach(println)
  }
}
