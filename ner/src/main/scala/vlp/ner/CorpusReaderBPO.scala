package vlp.ner

import scala.io.Source

import scala.util.parsing.json._
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.charset.StandardCharsets
import java.nio.file.StandardOpenOption
import vlp.tok.SentenceDetection

object CorpusReaderBPO {
  import scala.collection.JavaConversions._

  def preprocess(path: String): List[String] = {
    val lines = Source.fromFile(path, "UTF-8").getLines().toList
      .map(_.replaceAll("""\|""", "").replaceAll("""(\\t)+""", "").replaceAll("""_{3,}""", "").trim())
      .map(_.replaceAll("""\s{2,}""", " "))
      .filter(_.nonEmpty)
      .filter(line => !line.matches("\\-{5,}"))
    Files.write(Paths.get(path + ".json"), lines, StandardCharsets.UTF_8, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE) 
    lines
  }

  def selectBody(content: String): String = {
    val beginMarkers = Set("Kính gửi", "Kính gởi")
    val endMarkers = Set("Nơi nhận", "KT.")
    // find the max index of a begin marker
    val us = beginMarkers.map(marker => content.indexOf(marker))
    val u = if (us.nonEmpty) us.max else 0
    // find the min index of an end marker
    val vs = endMarkers.map(marker => content.indexOf(marker)).filter(_ >= 0)
    val v = if (vs.nonEmpty) vs.min else content.size
    content.slice(u, v)
  }

  def main(args: Array[String]): Unit = {
    val path = "dat/ner/bpo/99-5000-batch-01.txt"
    // preprocess(path)
    val content = Source.fromFile(path + ".json", "UTF-8").getLines().toList.mkString(" ")
    val documents = JSON.parseFull(content).get.asInstanceOf[List[Map[String, String]]]
    println(documents.size)
    // filter all sentences which have at leat one entity and has at least 20 characters
    val xs = documents.flatMap{ document => 
      val about = document("about")
      val body = selectBody(document("content"))
      val sents = SentenceDetection.run(body)
      List(about) ++ sents.filter(s => s.size >= 20 && s.contains("<") && s.contains("</")) ++ List("")
    }
    Files.write(Paths.get(path + ".sents"), xs, StandardCharsets.UTF_8, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
  }
}
