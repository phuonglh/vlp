package vlp.nli

import scala.util.parsing.json._
import org.apache.spark.rdd.RDD
import org.json4s.jackson.Serialization
import org.json4s._
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets

object XNLI {

    /**
     * Convert JSONL format to JSON format.
     */ 
    def convert(jsonlPath: String, jsonPath: String): Unit = {
        val s = scala.io.Source.fromFile(jsonlPath).getLines().toList
        val elements = s.map(x => JSON.parseFull(x).get.asInstanceOf[Map[String,Any]])
        
        println(elements.size)
        implicit val formats = Serialization.formats(NoTypeHints)
        val content = elements.map(e => Serialization.writePretty(e) + ",")
        val output = List("[") ++ content ++ List("]")
        import scala.collection.JavaConversions._
        Files.write(Paths.get(jsonPath), output, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    }

    def main(args: Array[String]): Unit = {
        val jsonlPath = "dat/nli/XNLI-1.0/vi.jsonl"
        val jsonPath = "dat/nli/XNLI-1.0/vi.json"
        convert(jsonlPath, jsonPath)
    }
}
