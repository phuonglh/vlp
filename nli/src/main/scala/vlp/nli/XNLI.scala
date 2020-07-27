package vlp.nli

import scala.util.parsing.json._
import org.apache.spark.rdd.RDD
import org.json4s.jackson.Serialization
import org.json4s._
import java.nio.file.{Paths, Files, StandardOpenOption}

object XNLI {
    def main(args: Array[String]): Unit = {
        val path = "dat/nli/xnli.dev.vi.jsonl"
        val s = scala.io.Source.fromFile(path).getLines().toList
        val elements = s.map(x => JSON.parseFull(x).get.asInstanceOf[Map[String,Any]])
        
        println(elements.size)
        implicit val formats = Serialization.formats(NoTypeHints)
        val content = Serialization.writePretty(elements)
        Files.write(Paths.get("dat/nli/xnli.dev.vi.json"), content.getBytes, StandardOpenOption.CREATE)        
    }
}
