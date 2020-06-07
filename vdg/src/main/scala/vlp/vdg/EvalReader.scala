package vlp.vdg

import org.json4s.DefaultFormats
import org.json4s._
import org.json4s.jackson.JsonMethods.parse

import scala.collection.mutable.ListBuffer
import scala.io.Source

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
    val j = 1

    val types = List("GRU", "LSTM")
    for (r <- types) {
      println(r)
      val grus = result.filter(_.dataPath.contains(corpus)).filter(_.recurrentType == r).filter(_.numLayers == j)
      val averageAccuracy = grus.groupBy(_.hiddenUnits).map { pair =>
        val k = pair._2.size
        (pair._1, pair._2.map(_.trainingScore).sum / k, pair._2.map(_.validationScore).sum / k, pair._2.map(_.testScore).sum / k)
      }.toList.sortBy(_._1)
      averageAccuracy.foreach(println)
      // print to the format of Tikz figure for LaTeX source
      averageAccuracy.map(t => (t._1, t._4*100)).foreach(e => print(s"(${e._1},${e._2}) "))
      println
    }
  }
}
