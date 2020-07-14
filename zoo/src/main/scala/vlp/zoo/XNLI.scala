package vlp.zoo

import scala.util.parsing.json._
import org.apache.spark.rdd.RDD

object XNLI {
    def main(args: Array[String]): Unit = {
        val path = "dat/nli/XNLI-1.0/xnli.dev.jsonl"
        val s = scala.io.Source.fromFile(path).getLines().toList
        println(s.size)
        val elements = s.par.map(x => JSON.parseFull(x).get.asInstanceOf[Map[String,Any]])
            .filter(map => map("language") == "vi")
        
        println(elements.size)
        elements.foreach(println)
    }
}
