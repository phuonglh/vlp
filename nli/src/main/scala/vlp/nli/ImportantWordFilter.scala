package vlp.nli

import scala.util.parsing.json._
import org.apache.spark.sql.SparkSession
import org.json4s.jackson.Serialization
import org.json4s._
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets
import scala.io.Source

/**
 *  This utility scans a part-of-speech tagged NLI corpus and filters "important" words using 
 *  special words, nouns, verbs and adjectives. This utility takes as input the output of PoSTagging.
 * 
 * */ 
object ImportantWordFilter {

  val negation = Set("không", "chẳng", "chưa")

  def scan(taggedCorpusPath: String, premise: Boolean = true) = {
    val lines = Source.fromFile(taggedCorpusPath, "UTF-8").getLines().toList
    val map = collection.mutable.Map[String, collection.mutable.Set[String]]()
    val elements = lines.map(line => JSON.parseFull(line).get.asInstanceOf[Map[String,Any]])
    val taggedSents = if (premise) {
      elements.map { obj => obj("sentence1_tagged").toString() }
    } else {
      elements.map { obj => obj("sentence2_tagged").toString() }
    }
    taggedSents.foreach { s =>
      val xs = s.split(" ")
      xs.foreach { x => 
        val j = x.indexOf('/')
        val (word, tag) = (x.substring(0, j), x.substring(j+1))
        val key = tag match {
          case "P" => "P"
          case "Np" => "Np"
          case "N" => "N"
          case "V" => "V"
          case "A" => "A"
          case _ => ""
        }
        if (key.nonEmpty) {
          val value = map.getOrElse(key, collection.mutable.Set[String]())
          value += word.toLowerCase()
          map += (key -> value)
        }
      }
    }
    map
  }

  def main(args: Array[String]): Unit = {
    val premise = false
    val inputPath = "dat/nli/XNLI-1.0/vi.tag.jsonl"
    val outputPath = if (premise) "dat/nli/XNLI-1.0/vi.tag.wordP.jsonl" else "dat/nli/XNLI-1.0/vi.tag.wordH.jsonl"
    val map = scan(inputPath, premise)
    map.foreach{ case (word, set) => println(word + " ==> " + set.size) }
    // write JSONL file
    import scala.collection.JavaConversions._
    implicit val formats = Serialization.formats(NoTypeHints)
    val content = Serialization.writePretty(map)
    Files.write(Paths.get(outputPath), content.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    println("Done")
  }
}
