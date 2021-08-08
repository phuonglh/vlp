package vlp.qas

import scala.util.parsing.json._
import org.apache.spark.rdd.RDD
import org.json4s.jackson.Serialization
import org.json4s._
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets

/**
  * An utility to convert JSON data files from MONRE to JSONL format. The output is a file "all.jsonl" containing all 
  * samples, which will be processed.
  */
object MONRE {
    def main(args: Array[String]): Unit = {
        implicit val formats = Serialization.formats(NoTypeHints)
        import scala.collection.JavaConversions._
        val folder = "C:/Users/phuonglh/vlp/dat/qas/"
        val files = Array("01", "02", "03", "04")
        val paths = files.map(file => folder + file + ".json")
        val samples = paths.flatMap { path =>
            val s = scala.io.Source.fromFile(path)("UTF-8").getLines().toList.mkString
            val elements = JSON.parseFull(s).get.asInstanceOf[Map[String,Any]].get("data").get.asInstanceOf[List[Map[String,Any]]]
            println(elements.size)
            elements.map(e => Serialization.write(e))
        }
        Files.write(Paths.get(folder + "all.jsonl"), samples.toList, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
        println("Done.")
    }
}
