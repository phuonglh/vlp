package vlp.nli

import org.json4s.DefaultFormats
import org.json4s._
import org.json4s.jackson.JsonMethods.parse

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Map
import scala.io.Source
import java.text.DecimalFormat

object ScoresSummary {
  val formatter = new DecimalFormat("##.####")

  def main(args: Array[String]): Unit = {
    val path = "dat/nli/scores.json"
    val content = Source.fromFile(path).getLines().toList.mkString(" ")
    implicit val formats = DefaultFormats
    val jsArray = parse(content)
    val result = ListBuffer[Scores]()
    for (js <- jsArray.children)
      result += js.extract[Scores]

    val n = 40
    val arch = "seq"
    val types = List("cnn", "gru")
    val embeddingSizes = Array(25, 50, 80, 100)
    val encoderSizes = Array(25, 50, 80, 100, 128, 150, 200, 256, 300)
    for (r <- types) {
      val mean = Map[(Int, Int), (Double, Double)]()
      val variance = Map[(Int, Int), (Double, Double)]()
      for (encoderSize <- encoderSizes) {
        val elements = result.filter(_.arch == arch).filter(_.encoder == r).filter(_.maxSequenceLength == n).filter(_.encoderSize == encoderSize)
        val averageAccuracy = elements.groupBy(_.embeddingSize).map { pair =>
          val k = pair._2.size
          (pair._1, pair._2.map(_.trainingScores.last).sum/k, pair._2.map(_.testScore).sum/k)
        }.toList.sortBy(_._1)
        for (j <- 0 until averageAccuracy.size) {
          mean += ((averageAccuracy(j)._1, encoderSize) -> (averageAccuracy(j)._2, averageAccuracy(j)._3))
        }
        val deviation = elements.groupBy(_.embeddingSize).map { pair =>
          val k = pair._2.size
          val trainSum = pair._2.map(_.trainingScores.last).sum.toDouble
          val trainStd = pair._2.map(score => (score.trainingScores.last - trainSum)*(score.trainingScores.last - trainSum)).sum/k
          val testSum = pair._2.map(_.testScore).sum.toDouble
          val testStd = pair._2.map(score => (score.testScore - testSum)*(score.testScore - testSum)).sum/k
          (pair._1, trainStd, testStd)
        }.toList.sortBy(_._1)
        for (j <- 0 until deviation.size) {
          variance += ((deviation(j)._1, encoderSize) -> (deviation(j)._2, deviation(j)._3))
        }
      }
      println(s"mean($r) = ")
      for (w <- embeddingSizes) {
        for (e <- encoderSizes) {
          print((w, e) + "->" + mean.getOrElse((w, e), (0, 0)) + ", ")
        }
        println()
      }
      println()
      println(s"variance($r) = ")
      for (w <- embeddingSizes) {
        for (e <- encoderSizes) {
          print((w, e) + "->" + variance.getOrElse((w, e), (0, 0)) + ", ")
        }
        println()
      }
      println()
    }
  }
}
