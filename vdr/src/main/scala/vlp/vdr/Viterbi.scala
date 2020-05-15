package vlp.vdr

import scala.collection.mutable.ListBuffer

/**
 * @author Phuong LE-HONG, phuonglh@gmail.com
 * <p>
 * Sep 18, 2016, 11:01:12 AM
 * <p>
 * Viterbi decoding algorithm which finds a max path on 
 * a score lattice whose sum of values are maximal.  
 *
 */
object Viterbi {
  
  /**
   * Finds the best path and its score on a lattice of scores of size 
   * K*N, where K is the number of labels and N is the number of steps. 
   * @param scores
   * @return a pair of the best path and its score.
   */
  def decode(scores: Array[Array[Double]]): (List[Int], Double) = {
    val numLabels = scores.length
    val n = scores(0).length
    // fill the score tabular t
    val t = Array.ofDim[Double](numLabels, n)
    // copy the first column
    for (k <- 0 until numLabels) 
      t(k)(0) = scores(k)(0)
    for (j <- 1 until n; i <- 0 until numLabels) {
      var maxVal = Double.NegativeInfinity
      for (k <- 0 until numLabels) 
        if (t(k)(j-1) > maxVal) maxVal = t(k)(j-1)
      t(i)(j) = maxVal + scores(i)(j)
    }
    // trace back on the lattice to find the best path
    val path = ListBuffer[Int]()
    // the last id of the path is the position with max value of the last column
    var maxVal = Double.NegativeInfinity
    var maxId = -1
    for (k <- 0 until numLabels) 
      if (t(k)(n-1) > maxVal) {
        maxVal = t(k)(n-1)
        maxId = k
      }
    path.append(maxId)
    // 
    for (j <- n-2 to 0 by -1) {
      val u = path(n-2-j)
      var maxId = -1
      for (k <- 0 until numLabels) {
        if ((t(k)(j) + scores(u)(j+1) - t(u)(j+1)).abs <= 1e-6)  // floating-point equality 
          maxId = k
      }
      path.append(maxId)
    }
    // return the result pair
    (path.toList.reverse, t(numLabels-1)(n-1))
  }
  
  /**
   * Finds the best path and its score on a lattice of scores of size 
   * N*K*K, where K is the number of labels and N is the number of steps. 
   * @param scores
   * @return a pair of the best path and its score.
   */
  def decode(scores: List[Array[Array[Double]]]): (List[Int], Double) = {
    val n = scores.length
    val numLabels = scores(0).length
    // fill the score tabular t
    val t = ListBuffer[Array[Array[Double]]]()
    t.append(scores(0))
    for (j <- 1 until n) {
      val s = scores(j)
      for (v <-0 until numLabels) {
        var maxV = Double.NegativeInfinity
        for (u <- 0 until numLabels) {
          val z = t(j-1)(u)(v) + s(u)(v) 
          if (z > maxV) {
            maxV = z
          }
        }
      }
      t.append(Array.ofDim[Double](numLabels, numLabels))
    }
    (Nil, 0d)
  }
  
  def main(args: Array[String]): Unit = {
    val scores1 = Array(Array(0.3, 0.1, 0.0), Array(0.2, 0.4, 0.1), Array(0.1, 0.4, 0.2));
    println(decode(scores1))
    val scores2 = Array(
        Array(29.0, 4, 20, 46, 30), 
				Array(13.0, 95, 52, 33, 56),
				Array(87.0, 25, 19, 50, 23), 
				Array(92.0, 28, 28, 45, 54),
				Array(30.0, 64, 25, 29, 80))
		println(decode(scores2))
		val scores3 = Array(
		    Array(0.321223, 0.726969, 0.802805, 0.747823, 0.848001),
				Array(0.509316, 0.330791, 0.186159, 0.800983, 0.320709),
				Array(0.078514, 0.110131, 0.688176, 0.080399, 0.823965),
				Array(0.602709, 0.017961, 0.162477, 0.962956, 0.897437),
				Array(0.208598, 0.232417, 0.286169, 0.865968, 0.181086))
		println(decode(scores3))
		val scores4 = Array(
		    Array(2.0, 0, 1, 0, 8, 0),
				Array(2.0, 3, 0, 2, 8, 5),
				Array(2.0, 2, 0, 7, 1, 2),
				Array(0.0, 9, 0, 9, 1, 5))
		println(decode(scores4))
  }
}