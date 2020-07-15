package vlp.vdg

import org.json4s.DefaultFormats
import org.json4s._
import org.json4s.jackson.JsonMethods.parse

import scala.collection.mutable.ListBuffer
import scala.io.Source
import java.text.DecimalFormat

/**
  * Reads JSON evaluation results and computes
  * the averaged scores for each configuration.
  */
object EvalReader {
  def main(args: Array[String]): Unit = {
    val inputPath = "dat/vdg/scores.json"
    val content = Source.fromFile(inputPath).getLines().toList.mkString(" ")
    implicit val formats = DefaultFormats
    val jsArray = parse(content)
    val result = ListBuffer[ConfigEval]()
    for (js <- jsArray.children)
      result += js.extract[ConfigEval]
    println(result.size)
    
    val corpus = "vtb.txt"
    val j = 2

    val formatter = new DecimalFormat("##.####")
    val types = List("GRU", "LSTM")
    for (r <- types) {
      println(r)
      val grus = result.filter(_.dataPath.contains(corpus)).filter(_.recurrentType == r).filter(_.numLayers == j)
      val averageAccuracy = grus.groupBy(_.hiddenUnits).map { pair =>
        val k = pair._2.size
        (pair._1, pair._2.map(_.trainingScore).sum/k, pair._2.map(_.validationScore).sum/k, pair._2.map(_.testScore).sum/k)
      }.toList.sortBy(_._1)
      averageAccuracy.foreach(println)
      // print to the format of Tikz figure for LaTeX source
      println("Average training accuracy: ")
      averageAccuracy.map(t => (t._1, t._2*100)).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
      println()
      println("Average dev. accuracy: ")
      averageAccuracy.map(t => (t._1, t._3*100)).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
      println()
      println("Average test accuracy: ")
      averageAccuracy.map(t => (t._1, t._4*100)).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
      // standard deviation
      val deviation = grus.groupBy(_.hiddenUnits).map { pair =>
        val k = pair._2.size
        val trainSum = pair._2.map(_.trainingScore).sum.toDouble
        val trainStd = pair._2.map(config => (config.trainingScore - trainSum)*(config.trainingScore - trainSum)).sum/k
        val validationSum = pair._2.map(_.validationScore).sum.toDouble
        val validationStd = pair._2.map(config => (config.validationScore - validationSum)*(config.validationScore - validationSum)).sum/k
        val testSum = pair._2.map(_.testScore).sum.toDouble
        val testStd = pair._2.map(config => (config.testScore - testSum)*(config.testScore - testSum)).sum/k
        (pair._1, trainStd, validationStd, testStd)
      }.toList.sortBy(_._1)
      println()
      println("std(training): ")
      deviation.map(t => (t._1, Math.sqrt(t._2))).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
      println()
      println("std(dev.): ")
      deviation.map(t => (t._1, Math.sqrt(t._3))).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
      println()
      println("std(test): ")
      deviation.map(t => (t._1, Math.sqrt(t._4))).foreach(e => print(s"(${formatter.format(e._1)},${formatter.format(e._2)}) "))
      println()
      println()
    }
  }
}
