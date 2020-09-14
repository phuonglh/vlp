package vlp.nli

import scalaj.http._
import java.io.BufferedReader
import java.nio.charset.StandardCharsets
import java.io.InputStreamReader
import java.util.stream.Collectors

import scala.util.parsing.json._
import org.json4s.jackson.Serialization
import org.json4s._
import org.json4s.jackson.JsonMethods._
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
    * @return a JSON string
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

  /**
   * Lookups a list of words in ConceptNet and write results to a JSONL file.
   * @param words
   * @param outputPath
   * 
   */
  def lookup(words: List[String], outputPath: String): Unit = {
    val results = new mutable.ListBuffer[String]()
    for (j <- 0 until words.size) {
      val json = lookup(words(j))
      results.append(json)
      Thread.sleep(510)
      if (j % 100 == 0)
        println(words(j))
    }
    import scala.collection.JavaConversions._
    Files.write(Paths.get(outputPath), results.toList, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
  }

  /**
   * Query all end information of a node given a relation.
   * @param node, for example "vào"
   * @param relation, for example "/r/Synonym"
   * @return a list of (word, language) such as [(vô,vi), (herein,de), (enter,en), (betreten,de)]
   * 
   */ 
  def query(node: String, relation: String): Seq[(String, String)] = {
    val json = parse(node) \ "edges"
    val result = mutable.ListBuffer[(String, String)]()
    for (child <- json.children) {
      val rel = (child \ "rel" \ "@id" \\ classOf[JString])(0)
      if (rel == relation) {
        val endNode = child \ "end"
        val label = (endNode \ "label" \\ classOf[JString])(0)
        val lang = (endNode \ "language" \\ classOf[JString])(0)
        result += ((label, lang))
      }
    }
    result.toSeq
  }

  def main(args: Array[String]): Unit = {
    val vocabulary = Source.fromFile("dat/nli/XNLI-1.0/vi.vocab.txt").getLines().toList
    val startIndex = if (args.size > 0) args(0).toInt else 0
    val endIndex = if (args.size > 1) args(1).toInt else vocabulary.size
    val selection = vocabulary.slice(startIndex, endIndex)
    lookup(selection, "dat/nli/ccn.json")
    println("Done.")

    // val node = Source.fromFile("dat/ccn/0.json").getLines().toList.mkString(" ")
    // val relations = Array("/r/Synonym", "/r/Antonym", "/r/IsA", "/r/PartOf", "/r/HasA", "/r/UsedFor", "/r/CapableOf", "/r/MadeOf", 
    //   "/r/DefinedAs", "/r/Causes", "/r/Desires", "/r/RelatedTo")
    // for (relation <- relations) {
    //   val result = query(node, relation)
    //   println(relation + " => " + result.mkString(" "))
    // }

    // println(lookup("vào", 40))
  }
}
