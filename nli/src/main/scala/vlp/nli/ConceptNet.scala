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
   * @param node, a JSON node of a word, for example "vào"
   * @param relation, for example "/r/Synonym"
   * @return a set of (word, language) such as [(vô,vi), (herein,de), (enter,en), (betreten,de)]
   * 
   */ 
  def query(node: String, relation: String): Set[(String, String)] = {
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
    result.toSet
  }

  /**
    * Query all end information of a node given a sequence of relations. 
    *
    * @param node
    * @param relations
    * @return a map of sets of (word, language) pairs.
    */
  def query(node: String, relations: Seq[String]): Map[String, Set[(String, String)]] = {
    val result = mutable.Map[String, Set[(String, String)]]()
    relations.foreach{ relation => 
      val xs = query(node, relation)
      result += (relation -> xs)
    }
    result.toMap
  }

  def query(nodes: Seq[String], relations: Seq[String], outputPath: String): Unit = {
    val result = nodes.map { node => 
      val id = (parse(node) \ "@id" \\ classOf[JString])(0)
      val word = id.substring(id.lastIndexOf('/') + 1)
      val xs = query(node, relations)
      val ys = relations.map { rel =>
        val pairs = xs(rel)
        if (pairs.nonEmpty) {
          pairs.map { p => 
            val u = p._1.replaceAll(" ", "_")
            val v = p._2
            rel + '/' + u + '/' + v
          }.mkString(" ")
        } else ""
      }.filter(_.nonEmpty)
      word + "\t" + ys.mkString(" ")
    }
    import scala.collection.JavaConversions._
    Files.write(Paths.get(outputPath), result, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def main(args: Array[String]): Unit = {
    // println(lookup("vào", 40))

    // query the ConceptNet server using al words in the Vietnamese vocab and 
    // write result to a JSON file.
    // val vocabulary = Source.fromFile("dat/nli/XNLI-1.0/vi.vocab.txt").getLines().toList
    // val startIndex = if (args.size > 0) args(0).toInt else 0
    // val endIndex = if (args.size > 1) args(1).toInt else vocabulary.size
    // val selection = vocabulary.slice(startIndex, endIndex)
    // lookup(selection, "dat/nli/ccn.json")

    // val node = Source.fromFile("dat/ccn/0.json").getLines().toList.mkString(" ")
    // val relations = Seq("/r/Synonym", "/r/Antonym", "/r/IsA", "/r/PartOf", "/r/HasA", "/r/UsedFor", "/r/CapableOf", "/r/MadeOf", 
    //   "/r/DefinedAs", "/r/Causes", "/r/Desires", "/r/RelatedTo")
    // val result = query(node, relations)
    // result.foreach(println)

    val nodes = Source.fromFile("dat/ccn/ccn.vi.json", "UTF-8").getLines().toList
    val relations = Seq("/r/Synonym", "/r/Antonym", "/r/IsA", "/r/PartOf", "/r/HasA", "/r/UsedFor", "/r/CapableOf", "/r/MadeOf", 
      "/r/DefinedAs", "/r/Causes", "/r/Desires", "/r/RelatedTo")
    query(nodes, relations, "dat/ccn/ccn.vi.tsv")
  }
}
