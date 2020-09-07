package vlp.nli

import org.json4s.DefaultFormats
import org.json4s._
import org.json4s.jackson.JsonMethods.parse

import scala.collection.mutable.ListBuffer
import scala.io.Source
import java.text.DecimalFormat

object ScoresSummary {
  val formatter = new DecimalFormat("##.####")

  def bow(elements: Seq[Scores]): Unit = {
      println("Number of elements = " + elements.size)
      val averageAccuracy = elements.groupBy(_.embeddingSize).map { pair =>
        val k = pair._2.size
        println(k)
        (pair._1, pair._2.map(_.trainingScores.last).sum/k, pair._2.map(_.testScore).sum/k)
      }.toList.sortBy(_._1)
      averageAccuracy.foreach(println)
      // print to the format of Tikz figure for LaTeX source
      println("Average training accuracy: ")
      averageAccuracy.map(t => (t._1, t._2*100)).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
      println()
      println("Average test. accuracy: ")
      averageAccuracy.map(t => (t._1, t._3*100)).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
      println()
      // standard deviation
      val deviation = elements.groupBy(_.embeddingSize).map { pair =>
        val k = pair._2.size
        val trainSum = pair._2.map(_.trainingScores.last).sum.toDouble
        val trainStd = pair._2.map(score => (score.trainingScores.last - trainSum)*(score.trainingScores.last - trainSum)).sum/k
        val testSum = pair._2.map(_.testScore).sum.toDouble
        val testStd = pair._2.map(score => (score.testScore - testSum)*(score.testScore - testSum)).sum/k
        (pair._1, trainStd, testStd)
      }.toList.sortBy(_._1)
      println()
      println("std(training): ")
      deviation.map(t => (t._1, Math.sqrt(t._2))).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
      println()
      println("std(test): ")
      deviation.map(t => (t._1, Math.sqrt(t._3))).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
      println()
      println()
  }

  def main(args: Array[String]): Unit = {
    val path = "dat/nli/scores.nli.json"
    val content = Source.fromFile(path).getLines().toList.mkString(" ")
    implicit val formats = DefaultFormats
    val jsArray = parse(content)
    val result = ListBuffer[Scores]()
    for (js <- jsArray.children)
      result += js.extract[Scores]
    println(result.size)

    val n = 40
    val arch = "bow"
    if (arch == "bow") {
      val elements = result.filter(_.arch == arch).filter(_.maxSequenceLength == n)
      bow(elements)
    } else {
      val encoderSize = 25
      val types = List("cnn", "gru")
      for (r <- types) {
        println(r)
        val elements = result.filter(_.arch == arch).filter(_.encoder == r).filter(_.maxSequenceLength == n).filter(_.encoderSize == encoderSize)
        println("Number of elements = " + elements.size)
        val averageAccuracy = elements.groupBy(_.embeddingSize).map { pair =>
          val k = pair._2.size
          println(k)
          (pair._1, pair._2.map(_.trainingScores.last).sum/k, pair._2.map(_.testScore).sum/k)
        }.toList.sortBy(_._1)
        averageAccuracy.foreach(println)
        // print to the format of Tikz figure for LaTeX source
        println("Average training accuracy: ")
        averageAccuracy.map(t => (t._1, t._2*100)).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
        println()
        println("Average test. accuracy: ")
        averageAccuracy.map(t => (t._1, t._3*100)).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
        println()
        // standard deviation
        val deviation = elements.groupBy(_.embeddingSize).map { pair =>
          val k = pair._2.size
          val trainSum = pair._2.map(_.trainingScores.last).sum.toDouble
          val trainStd = pair._2.map(score => (score.trainingScores.last - trainSum)*(score.trainingScores.last - trainSum)).sum/k
          val testSum = pair._2.map(_.testScore).sum.toDouble
          val testStd = pair._2.map(score => (score.testScore - testSum)*(score.testScore - testSum)).sum/k
          (pair._1, trainStd, testStd)
        }.toList.sortBy(_._1)
        println()
        println("std(training): ")
        deviation.map(t => (t._1, Math.sqrt(t._2))).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
        println()
        println("std(test): ")
        deviation.map(t => (t._1, Math.sqrt(t._3))).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
        println()
        println()
      }
    }
  }
}
