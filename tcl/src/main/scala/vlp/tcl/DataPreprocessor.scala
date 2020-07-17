package vlp.tcl

import scala.io.Source
import scala.collection.mutable.ListBuffer
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.charset.StandardCharsets

/**
  * A utility for the module: convert original datasets (training) to 
  * a TSV file of format "category <tab> text". 
  * 
  */
object DataPreprocessor {
  def main(args: Array[String]): Unit = {
    // build the labels map
    val ys = Source.fromFile("dat/hsd/vlsp/03_train_label.csv", "UTF-8").getLines().toList.filter(_.trim.nonEmpty)
    val labelMap = ys.map(line => {
      val i = line.indexOf(',')
      (line.substring(0, i), line.substring(i+1))
    }).toMap

    val lines = Source.fromFile("dat/hsd/vlsp/02_train_text.csv", "UTF-8").getLines().toList.filter(_.trim.nonEmpty)
    // find the "train_" lines which separate samples
    val js = lines.zipWithIndex.filter(p => p._1.startsWith("train_"))
      .map(p => p._2)
    val samples = ListBuffer[Document]()
    var k = 0
    while (k < js.length - 1) {
      val s = lines.slice(js(k), js(k+1))
      val first = lines(js(k))
      val i = first.indexOf(',')
      val label = first.substring(0, i)
      val text = first.substring(i+1) + s.tail.mkString
      samples += Document(labelMap(label), text.filterNot(_ == '"'), "NA")
      k = k + 1
    }
    samples.takeRight(10).foreach(println)
    // remove clean (0) documents which has a length < 20
    val filteredSamples = samples.filter(document => 
      document.category != "0" || (document.category == "0" && document.text.size > 20)
    )
    filteredSamples.groupBy(document => document.category).foreach(p => 
      println(p._1 + " ==> " + p._2.size) 
    )
    // write out the preprocessed data to a tsv file.
    val output = filteredSamples.map(document => document.category + "\t" + document.text)
    import scala.collection.JavaConversions._
    Files.write(Paths.get("dat/hsd/train.tsv"), output, StandardCharsets.UTF_8)
  }
}
