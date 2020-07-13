package vlp.zoo

import scala.util.parsing.json._
import java.text.DecimalFormat
/**
  * Reads validation scores in the TensorFlow format to display in Tikz figure.
  */
object ScoreReaders {

  def main(args: Array[String]): Unit = {
    val formatter = new DecimalFormat("##.####")
    val paths = Array("/Users/phuonglh/vlp/dat/zoo/tcl/run-cnn_2020-07-08.225442_validation-tag-Top1Accuracy.json",
      "/Users/phuonglh/vlp/dat/zoo/tcl/run-gru_2020-07-09.004432_validation-tag-Top1Accuracy.json",
      "/Users/phuonglh/vlp/dat/zoo/tcl/run-lstm_2020-07-09.084542_validation-tag-Top1Accuracy.json")
    paths.foreach { path =>   
      val s = scala.io.Source.fromFile(path).getLines().toList.head
      val triples = JSON.parseFull(s).get.asInstanceOf[List[Any]]
      var epoch = 1
      triples.foreach { t => 
        val List(a, b, c) = t
        print(s"($epoch, ${formatter.format(c)}) ")
        epoch = epoch + 1
      }
      println("\n")
    }
  }
}
