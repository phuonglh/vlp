package vlp.idx

import scala.io.Source
import org.json4s.jackson.Serialization
import org.json4s._
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets


case class Sheet(category: String, urls: List[String])
/**
  * Takes as input the output of [[NewsCategorizer]] and merges the URLs of the same category.
  * This will help create a corpus for text categorization.
  * phuonglh, July 2020.
  */
object CategoryMerger {

  final val cats = Map[String, String](
    "tai-chinh" -> "finance", "taichinh" -> "finance",
    "doanh-nghiep" -> "enterprise", "doanhnghiep" -> "enterprise", "doanh-nghiep/" -> "enterprise", "doanh-nghiep-d3" -> "enterprise",
    "chung-khoan" -> "stock", "chung-khoan/" -> "stock",
    "dau-tu" -> "investment", "dau-tu-d2" -> "investment",
    "bao-hiem/" -> "insurance",
    "hang-hoa" -> "commodity",
    "bat-dong-san" -> "realEstate",
    "tien-te/" -> "money",
    "ngan-hang-d5" -> "banking",
    "thoi-su-d1" -> "news", "quoc-te-d54" -> "news", "quoc-te" -> "news", "the-gioi" -> "news", "thegioi" -> "news",
    "thuong-mai-dien-tu" -> "business", "kinhdoanh" -> "business", "thuong-truong/" -> "business"
  )

  def main(args: Array[String]): Unit = {
    val lines = Source.fromFile("/Users/phuonglh/vlp/dat/idx/urls.txt").getLines().toList
    val xs = lines.map(line => {
      val parts = line.trim.split("\\s+")
      if (parts.size > 2) {
        val category = cats.getOrElse(parts(1), "other")
        val urls = parts.slice(2, cats.size).toList
        Sheet(category, urls)
      } else Sheet("", List.empty)
    }).filter(_.category.nonEmpty)
    val sheets = xs.groupBy(_.category).mapValues(list => list.flatMap(_.urls).toSet)
    implicit val formats = Serialization.formats(NoTypeHints)
    val content = Serialization.writePretty(sheets)
    Files.write(Paths.get("/Users/phuonglh/vlp/dat/idx/urls.json"), content.getBytes, StandardOpenOption.CREATE)
    // some statistics
    val stats = sheets.map(p => (p._1, p._2.size)).toList.sortBy(_._2)
    stats.foreach(println)
    println("Number of documents = " + stats.map(_._2).sum)
  }
}
