package vlp.vio

import org.jsoup.Jsoup
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

import scala.io.Source
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.charset.StandardCharsets
import java.nio.file.StandardOpenOption

/**
  * phuonglh, August, 2020.
  * 
  * Import questions in a JSON file which is exported by VioEdu and clean HTML tags, clean and export
  * them to a pretty format. 
  * 
  */
object QuestionCleaner {

  def main(args: Array[String]): Unit = {
    val lines = Source.fromFile("dat/vio/edutech_dev.questions.json").getLines().toList.filterNot(_.contains("s3.vio.edu.vn"))
    println("Number of valid questions = " + lines.size)
    implicit val formats = DefaultFormats
    val elements = lines.map(line => {
      val json = parse(line)
      val id = (json \ "_id" \ "$oid").extract[String]
      val content = (json \ "content").extract[String]
      val xs = (json \ "answers")
      val us = (xs \ "text").extract[List[String]]
      val vs = (xs \ "correct").extract[List[Boolean]]
      val answers = us.zip(vs).map{case (u, v) => Answer(Jsoup.parse(u).text(), v)}
      val qa = QA(id, Jsoup.parse(content).text(), answers)
      Serialization.writePretty(qa)
    })
    import scala.collection.JavaConversions._
    val outputPath = "dat/vio/edutech.grade6.json"
    val outputSt = "[" + elements.map(e => e).mkString(",\n") + "]"
    Files.write(Paths.get(outputPath), outputSt.getBytes, StandardOpenOption.CREATE)
    println(s"Result is written to ${outputPath}")
  }
}
