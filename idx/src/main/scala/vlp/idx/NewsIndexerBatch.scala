package vlp.idx

import scala.collection.JavaConversions._
import java.util.Date
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization
import org.json4s.NoTypeHints
import scala.io.Source
import java.nio.charset.StandardCharsets
import java.nio.charset.Charset
import java.nio.file.{Files, Paths}

/**
  * phuonglh, May 3, 2020.
  * 
  * Reads the MySQL database of URLs, extracts their contents and saves them to a file.
  */
object NewsIndexerBatch {
  final val batchSize = 20000

  def main(args: Array[String]): Unit = {
    val urls = MySQL.getURLs
    System.setProperty("http.agent", "Chrome")
    System.setProperty("https.protocols", "TLSv1,TLSv1.1,TLSv1.2")
    println(s"#(totalURLs) = ${urls.size}")
    val batches = urls.sliding(batchSize)
    var i = 1
    for (batch <- batches) {
      println(batch.size)
      val news = batch.par.filterNot(u => u.contains("bbc.com") || u.contains("baohaiquan.vn")).map(url => {
        val content = NewsIndexer.extract(url)
        new News(url, content, new Date())
      }).toList
      val accept = (s: String) => (s.size >= 500 && !s.contains("<div") && !s.contains("<table") && !s.contains("</p>"))
      val ns = news.filter(x => accept(x.getContent()))
      println(s"#(batchSize) = ${ns.size}")
      implicit val formats = DefaultFormats
      implicit val f = Serialization.formats(NoTypeHints)
      val xs = ns.par.map(e => Serialization.write(e)).toList
      Files.write(Paths.get("dat/txt/" + i + ".json"), xs, StandardCharsets.UTF_8)
      i = i + 1
    }
  }
}