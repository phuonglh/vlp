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

    /**
      * Tokenizes XNLI corpus using a Vietnamese tokenizer.
      * 
      * @param jsonlPath
      * @param jsonPath
      */
    def tokenize(jsonlPath: String, jsonPath: String): Unit = {
        val s = scala.io.Source.fromFile(jsonlPath).getLines().toList
        val elements = s.map(x => JSON.parseFull(x).get.asInstanceOf[Map[String,Any]])

        val tokElements = elements.par.map { element => 
            val premise = element("sentence1_tokenized").toString()
            val hypothesis = element("sentence2_tokenized").toString()
            val premiseTokenized = vlp.tok.Tokenizer.tokenize(premise).map(_._3).mkString(" ")
            val hypothesisTokenized = vlp.tok.Tokenizer.tokenize(hypothesis).map(_._3).mkString(" ")
            Map("gold_label" -> element("gold_label"), "sentence1_tokenized" -> premiseTokenized, "sentence2_tokenized" -> hypothesisTokenized, 
                "both" -> (premiseTokenized + " | " + hypothesisTokenized))
        }.toList
        println(tokElements.size)
        implicit val formats = Serialization.formats(NoTypeHints)
        val content = tokElements.map(e => Serialization.write(e))
        import scala.collection.JavaConversions._
        Files.write(Paths.get(jsonPath), content, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)

    }

    def main(args: Array[String]): Unit = {
        val jsonlPath = "dat/nli/XNLI-1.0/vi.jsonl"
        val jsonPath = "dat/nli/XNLI-1.0/vi.json"
        // convert(jsonlPath, jsonPath)
        tokenize(jsonlPath, "dat/nli/XNLI-1.0/vi.tok.json")
        println("Done.")
    }
}
