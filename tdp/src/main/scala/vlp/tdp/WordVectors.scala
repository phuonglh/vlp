package vlp.tdp

import scala.io.Source

/**
  * Created by phuonglh on 7/3/17.
  * 
  * Distributed word vectors reader.
  */
object WordVectors {

  val lexicalDimension = 50
  val syntacticDimension = 10
  
  private def stringToDouble(str: String): Option[Double] = {
    import scala.util.control.Exception._
    catching(classOf[NumberFormatException]) opt str.toDouble
  }

  /**
    * Reads in a map of word -> word vector from a data path.
    * @param path
    * @return a vector
    */
  def read(path: String): Map[String, Vector[Double]] = {
    val lines = Source.fromFile(path, "UTF-8").getLines().filter(!_.isEmpty).toArray
    lines.map(line => {
      val a = line.split("\\s+")
      val w = a(0)
      val v = a.tail.map(e => stringToDouble(e).get).toVector
      (w, v)
    }).toMap
  }
}
