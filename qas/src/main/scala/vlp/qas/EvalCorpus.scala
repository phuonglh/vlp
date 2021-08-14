package vlp.qas

import scala.util.parsing.json._
import org.apache.spark.rdd.RDD
import org.json4s.jackson.Serialization
import org.json4s._
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets
import scala.collection.mutable.ArrayBuffer

case class EvalSample(
    id: String,
    question: String,
    keywords: List[String],
    rankedIds: List[String]
)

/**
 *  Reads eval. data and builds evaluation samples.
  * phuonglh, August 9, 2021.
  */
object EvalCorpus {
    def read(path: String): List[EvalSample] = {
        implicit val formats = Serialization.formats(NoTypeHints)
        import scala.collection.JavaConversions._
        val content = scala.io.Source.fromFile(path)("UTF-8").getLines().mkString
        val elements = JSON.parseFull(content).get.asInstanceOf[List[Map[String,Any]]]
        val samples = elements.map { element => 
            val xs = element.get("keyword").get.asInstanceOf[List[Any]]
            val keywords = new ArrayBuffer[String]()
            for (x <- xs) {
                x match {
                    case List(ws) => keywords ++= ws.asInstanceOf[List[String]]
                    case w => keywords += w.toString
                }
            }
            val ys = element.getOrElse("rankIdxs", List.empty[Any]).asInstanceOf[List[Any]]
            val rankedIds = new ArrayBuffer[String]()
            for (y <- ys) {
                val s = y.asInstanceOf[List[Any]].head.toString
                val j = s.indexOf(":")
                rankedIds += s.substring(j+1)
            }
            EvalSample(element.get("id").get.toString, element.get("question").get.toString,
                keywords.toList, rankedIds.toList
            )
        }
        return samples
    }

    def main(args: Array[String]): Unit = {
        val samples = read("C:/Users/phuonglh/vlp/dat/qas/result.json")
        samples.take(10).foreach(println)
        println(samples.size)
        println("Done.")
    }
    
}
