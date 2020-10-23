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

  def firstThreeArch(n:Int = 40, arch: String = "seq", result: ListBuffer[Scores]): Unit = {
    val types = List("cnn", "gru")
    val embeddingSizes = Array(25, 50, 80, 100)
    val encoderSizes = Array(100, 128, 150, 200, 256, 300)
    for (r <- types) {
      val mean = Map[(Int, Int), (Double, Double)]()
      val variance = Map[(Int, Int), (Double, Double)]()
      for (encoderSize <- encoderSizes) {
        val elements = result.filter(e => e.arch == arch && e.encoder == r && e.maxSequenceLength == n && e.encoderSize == encoderSize)
        val averageAccuracy = elements.groupBy(_.embeddingSize).map { pair =>
           val k = Math.min(pair._2.size, 3)
          (pair._1, pair._2.map(_.trainingScores.last).takeRight(k).sum/k, pair._2.map(_.testScore).sorted.takeRight(k).sum/k)
        }.toList.sortBy(_._1)
        for (j <- 0 until averageAccuracy.size) {
          mean += ((averageAccuracy(j)._1, encoderSize) -> (averageAccuracy(j)._2, averageAccuracy(j)._3))
        }
        val deviation = elements.groupBy(_.embeddingSize).map { pair =>
          val k = pair._2.size
          val trainSum = pair._2.map(_.trainingScores.last).sum.toDouble/k
          val trainStd = pair._2.map(score => (score.trainingScores.last - trainSum)*(score.trainingScores.last - trainSum)).sum/k
          val testSum = pair._2.map(_.testScore).sum.toDouble/k
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
          val value = mean.getOrElse((w, e), (0d, 0d))
          val s = (formatter.format(value._1), formatter.format(value._2))
          print((w, e) + "->" + s + ", ")
        }
        println()
      }
      println()
      println(s"variance($r) = ")
      for (w <- embeddingSizes) {
        for (e <- encoderSizes) {
          val value = variance.getOrElse((w, e), (0d, 0d))
          val s = (formatter.format(value._1), formatter.format(value._2))
          print((w, e) + "->" + s + ", ")
        }
        println()
      }
      println()
    }
  }

  def transformers(n: Int = 40, result: ListBuffer[Scores]): Unit = {
    val encoderSizes = Array(8, 16, 32, 48, 64, 80, 128, 160, 200, 256, 304)
    val mean = Map[Int, Double]()
    val std = Map[Int, Double]()
    for (encoderSize <- encoderSizes) {
      val elements = result.filter(e => e.arch == "trs" && e.maxSequenceLength == n && e.encoderSize == encoderSize)
      val k = elements.size
      val testAvg = elements.map { e => e.testScore }.sum/k
      val testVar = elements.map(e => (e.testScore - testAvg)*(e.testScore - testAvg)).sum/k
      mean += (encoderSize -> testAvg)
      std += (encoderSize -> Math.sqrt(testVar))
    }
    println("encoderSize\tmean\tstd")
    for (e <- encoderSizes) {
      println(e.toString + '\t' + mean.getOrElse(e, 0d) + '\t' + std.getOrElse(e, 0d))
    }
  }

  def main(args: Array[String]): Unit = {
    // val path = "dat/nli/scores.par.json"
    // val path = "dat/nli/scores.trs.x2.syllable.json"
    // val path = "dat/nli/scores.sem.gru.json"
    val path = "dat/nli/scores.en.json"

    val content = Source.fromFile(path).getLines().toList.mkString(" ")
    implicit val formats = DefaultFormats
    val jsArray = parse(content)
    val result = ListBuffer[Scores]()
    for (js <- jsArray.children)
      result += js.extract[Scores]
    result.take(3).foreach(println)

    // transformers(n, result)
    firstThreeArch(40, "seq", result)
    firstThreeArch(40, "par", result)
  }
}
