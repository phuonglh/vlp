package vlp.nli

import scalaj.http._
import java.io.BufferedReader
import java.nio.charset.StandardCharsets
import java.io.InputStreamReader
import java.util.stream.Collectors

import scala.util.parsing.json._
import org.json4s.jackson.Serialization
import org.json4s._
import scala.io.Source
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import scala.collection.mutable


/**
  * Lookup utility from ConceptNet API.
  * 
  */
object ConceptNet {
  val path = "http://api.conceptnet.io/c/vi/"
  implicit val formats = Serialization.formats(NoTypeHints)

  /**
    * Lookups a word in ConceptNet and returns a JSON string representation 
    * of the result.
    * @param word
    * @return
    */
  def lookup(word: String, offset: Int = 0): String = {
    val response: HttpResponse[String] = Http(path + word).param("offset", offset.toString)
      .header("Accept", "application/json")
      .header("Charset", "UTF-8")
      .execute(parser = { inputStream => {
        val reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))
        val text = reader.lines().collect(Collectors.joining("\n"))
        text
      }})
    val result = JSON.parseFull(response.body).get.asInstanceOf[Map[String,Any]]
    Serialization.write(result)
  }

  def main(args: Array[String]): Unit = {
    val vocabulary = Source.fromFile("dat/nli/XNLI-1.0/vi.vocab.txt").getLines().toList
    val startIndex = if (args.size > 0) args(0).toInt else 0
    val endIndex = if (args.size > 1) args(1).toInt else vocabulary.size
    val selection = vocabulary.slice(startIndex, endIndex)
    val results = new mutable.ListBuffer[String]()
    for (j <- 0 until selection.size) {
      val json = lookup(selection(j))
      results.append(json)
      Thread.sleep(510)
      if (j % 100 == 0)
        println(selection(j))
    }
    import scala.collection.JavaConversions._
    Files.write(Paths.get("dat/nli/ccn.json"), results.toList, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    println("Done.")
  }
}
