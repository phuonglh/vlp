package vlp.idx

import scala.collection._
import java.text.SimpleDateFormat
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.charset.StandardCharsets
import java.nio.file.StandardOpenOption

import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization
import org.json4s.NoTypeHints


case class Document(site: String, category: String, url: String, text: String)

object NewsCategorizer {

  def youth(date: String): List[(String, String, Set[String])] = {
    val categories = Array("tai-chinh", "doanh-nghiep", "mua-sam", "dau-tu")
    val urls = mutable.ListBuffer[(String, String, Set[String])]()
    for (category <- categories) {
      val set = NewsIndexer.extractURLs("https://tuoitre.vn", "kinh-doanh/" + category + ".htm", "/[\\p{Alnum}/-]+(\\d{4,})\\.htm", (s: String) => s.contains(date) && !s.contains("?"))
      urls += (("tuoitre.vn", category, set))
    }
    urls.toList
  }

  def saigonTimes: List[(String, String, Set[String])] = {
    val categories = Array("kinhdoanh", "taichinh", "doanhnghiep", "diendan", "thegioi")
    val urls = mutable.ListBuffer[(String, String, Set[String])]()
    for (category <- categories) {
      val set = NewsIndexer.extractURLs("https://www.thesaigontimes.vn", category, "/[\\p{Alnum}/-]+\\.html", (s: String) => !s.startsWith("/SaiGonTimes"))
      urls += (("www.thesaigontimes.vn", category, set))
    }
    urls.toList
  }

  def vnExpress: List[(String, String, Set[String])] = {
    val categories = Array("doanh-nghiep", "bat-dong-san", "thuong-mai-dien-tu", "hang-hoa", "chung-khoan", "quoc-te")
    val urls = mutable.ListBuffer[(String, String, Set[String])]()
    for (category <- categories) {
      val set = NewsIndexer.extractURLs("https://vnexpress.net", "kinh-doanh/" + category, "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (s: String) => s.contains(category))
      urls += (("vnexpress.net", category, set))
    }
    urls.toList
  }

  def fastStockNews: List[(String, String, Set[String])] = {
    val categories = Array("chung-khoan/", "thuong-truong/", "bao-hiem/", "doanh-nghiep/", "tien-te/")
    val urls = mutable.ListBuffer[(String, String, Set[String])]()
    for (category <- categories) {
      val set = NewsIndexer.extractURLs("https://tinnhanhchungkhoan.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.html",
        s => s.contains("-") && !s.contains("/toa-soan/") && !s.contains("/dai-hoi-co-dong/") && !s.contains("/don-doc/") && !s.contains("/cuoc-song/"))
        urls += (("tinhnhanhchungkhoan.vn", category, set))
    }
    urls.toList
  }

  def vir: List[(String, String, Set[String])] = {
    val urls = mutable.ListBuffer[(String, String, Set[String])]()
    val categories = Set("thoi-su-d1", "dau-tu-d2", "quoc-te-d54", "doanh-nghiep-d3", "doanh-nhan-d4", "ngan-hang-d5", "tai-chinh-chung-khoan-d6")
    for (category <- categories) {
      val set = NewsIndexer.extractURLs("https://baodautu.vn/" + category, "", "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (s: String) => true)
      urls += (("baodautu.vn", category, set))
    }
    urls.toList
  }

  def main(args: Array[String]): Unit = {
    val outputPath = args(0) 
    println(s"Output path = ${outputPath}")
    val dateFormat = new SimpleDateFormat("yyyyMMdd")
    val currentDate = dateFormat.format(new java.util.Date())
    val result = mutable.ListBuffer[(String, String, Set[String])]()
    println("Youth")
    result ++= youth(currentDate)
    println("The Saigontimes")
    result ++= saigonTimes
    println("vnExpress")
    result ++= vnExpress
    println("Fast stock")
    result ++= fastStockNews
    println("VIR")
    result ++= vir

    val output = result.map{case (a, b, c) => a + " " + b + " " + c.mkString(" ")}
    import scala.collection.JavaConversions._
    Files.write(Paths.get(outputPath + "/vlp/dat/idx/urls.txt"), output.toList, StandardOpenOption.APPEND)

    val texts = result.flatMap(t => t._3.par.map(x => Document(t._1, t._2, x, NewsIndexer.extract(x))))
    implicit val formats = DefaultFormats
    implicit val f = Serialization.formats(NoTypeHints)
    val xs = texts.par.map(e => Serialization.write(e)).toList
    println("Writing extraction result...")
    Files.write(Paths.get(outputPath + "/vlp/dat/idx/texts.json"), xs, StandardCharsets.UTF_8)
  }
}
