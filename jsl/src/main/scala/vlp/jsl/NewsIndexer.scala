package vlp.jsl

import java.io.IOException
import java.net.{MalformedURLException, URL}
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util.Date
import java.util.regex.Pattern

import de.l3s.boilerpipe.BoilerpipeProcessingException
import de.l3s.boilerpipe.extractors.ArticleExtractor
import org.apache.commons.io.IOUtils
import org.xml.sax.InputSource

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import org.apache.log4j.{Level, Logger}
import org.slf4j.LoggerFactory

import org.json4s.jackson.Serialization
import org.json4s._
import java.nio.file.{Paths, Files, StandardOpenOption}

import scala.concurrent.Await
import scala.concurrent.Future
import scala.concurrent.ExecutionContext
import java.util.concurrent.TimeoutException
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import org.apache.kafka.clients.producer.ProducerRecord

case class Page(url: String, content: String, date: Date)

/**
  * Extractor of health news content for a given day from some large online newswire. 
  * The news are sent to a Apache Kafka server.
  *
  */
object NewsIndexer {
  final val logger = LoggerFactory.getLogger(getClass.getName)
  /**
    * Extracts the main content of a news URL.
    *
    * @param url a URL to extract text.
    * @return main content in string.
    */
  def extract(url: String): String = {
    try {
      val path = new URL(url)
      val is = new InputSource
      is.setEncoding("UTF-8")
      is.setByteStream(path.openStream)
      ArticleExtractor.INSTANCE.getText(is)
    } catch {
      case e: MalformedURLException => logger.error(e.getMessage); ""
      case e: BoilerpipeProcessingException => logger.error(e.getMessage); ""
      case e: IOException => logger.error(e.getMessage); ""
      case _: Exception => logger.error("Other exception"); ""
    }
  }

  /**
    * Extracts URLs of a given category of a given site.
    * @param site a site
    * @param category a category in that site
    * @param regexp a regular expression to get url
    * @param filter a URL filter
    * @return a set of URLs.
    */
  def extractURLs(site: String, category: String, regexp: String, filter: String => Boolean): Set[String] = {
    val urls = mutable.Set[String]()
    try {
      val address = site + "/" + category
      val connection = new URL(address).openConnection
      connection.setReadTimeout(5000)
      val inputStream = connection.getInputStream
      val pageContent = IOUtils.toString(inputStream, StandardCharsets.UTF_8)
      val lines = pageContent.split("\\n")
      val pattern = Pattern.compile(regexp)
      for (line <- lines) {
        val matcher = pattern.matcher(line)
        while (matcher.find()) {
          val url = matcher.group()
          if (filter(url)) {
            urls += (site + url)
          }
        }
      }
    } catch {
      case e: IOException => e.printStackTrace()
      case e: Exception => 
        logger.error(site + "/" + category)
        e.printStackTrace()
    } 
    urls.toSet
  }

  def vnExpress: Set[String] = {
    val urls = mutable.Set[String]()
    val categories = List("kinh-doanh", "quoc-te", "doanh-nghiep", "chung-khoan", "bat-dong-san", "ebank", "vi-mo", "tien-cua-toi", "hang-hoa", "bao-hiem")
    categories.foreach { category =>
      urls ++= extractURLs("https://vnexpress.net", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (s: String) => true)
    }
    logger.info("vnExpress.vn/kinh-doanh => " + urls.size)
    urls.toSet
  }

  def vtv(date: String): Set[String] = {
    val urls = mutable.Set[String]()
    val categories = List("bat-dong-san.htm", "tai-chinh.htm", "thi-truong.htm", "goc-doanh-nghiep.htm")
    categories.foreach { category => 
      urls ++= extractURLs("https://vtv.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.htm", (s: String) => s.contains(date))
    }
    logger.info("suckhoe.vtv.vn => " + urls.size)
    urls.toSet
  }

  def youth(date: String): Set[String] = {
    val urls = mutable.Set[String]()
    val categories = List("tai-chinh.htm", "doanh-nghiep.htm", "mua-sam.htm", "dau-tu.htm")
    categories.foreach { category => 
      urls ++= extractURLs("https://tuoitre.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.htm", (s: String) => s.contains(date) && !s.contains("?"))
    }
    logger.info("tuoitre.vn => " + urls.size)
    urls.toSet
  }
  
  def vietnamnet: Set[String] = {
    val urls = mutable.Set[String]()
    val categories = List("kinh-doanh/tai-chinh", "kinh-doanh/dau-tu", "kinh-doanh/thi-truong", "kinh-doanh/doanh-nhan", "kinh-doanh/tu-van-tai-chinh")
    categories.foreach { category => 
      urls ++= extractURLs("https://vietnamnet.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (s: String) => s.contains("-"))
    }
    urls.toSet.filterNot(_.contains("/en/"))
  }

  def sggp: Set[String] = {
    val urls = mutable.Set[String]()
    val categories = List("thitruongkt/", "xaydungdiaoc/", "nganhangchungkhoan/", "nongnghiepkt/", "thong-tin-kinh-te/", "dautukt/")
    categories.foreach { category =>
      urls ++= extractURLs("https://www.sggp.org.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (s: String) => s.contains("-"))
    }
    logger.info("sggp.org.vn => " + urls.size)
    urls.toSet
  }

  def pioneer: Set[String] = {
    val urls = mutable.Set[String]()
    val categories = List("kinh-te-doanh-nghiep/", "kinh-te-doanh-nhan/", "kinh-te-chung-khoan/", "do-thi/", "thi-truong/", "nha-dep/")
    categories.foreach { category => 
      urls ++= extractURLs("https://www.tienphong.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.tpo", (s: String) => s.contains("-"))
    }
    logger.info("tienphong.vn => " + urls.size)
    urls.toSet
  }

  def runWithTimeout[T](timeout: Long)(f: => T): Option[T] = {
    try {
      Some(Await.result(Future(f), timeout.seconds))
    } catch {
      case e: TimeoutException => None
    }
  }

  def run(date: String): Unit = {
    System.setProperty("http.agent", "Chrome")
    System.setProperty("https.protocols", "TLSv1,TLSv1.1,TLSv1.2")

    import scala.collection.JavaConversions._
    val urls = mutable.Set[String]()
    urls ++= vnExpress
    urls ++= vtv(date)
    urls ++= youth(date)
    urls ++= vietnamnet
    urls ++= pioneer
    urls ++= sggp

    logger.info(s"#(totalURLs) = ${urls.size}")

    val kafkaProducer = Kafka.createProducer(Kafka.SERVERS)
    val news = urls.par.map(url => {
      logger.info(url)
      val content = runWithTimeout(5000)(extract(url)).get
      if (content.size >= 500 && !content.contains("div") && !content.contains("class=") && !content.contains("script")) {
        kafkaProducer.send(new ProducerRecord[String, String](Kafka.GROUP_ID, url, content))
        Page(url, content, new Date())
      } else {
        Page(url, "", new Date())
      }
    }).toList
    kafkaProducer.close()

    if (news.nonEmpty) {
      implicit val formats = Serialization.formats(NoTypeHints)
      val content = Serialization.writePretty(news)
      Files.write(Paths.get(System.getProperty("user.dir"), "dat", date + ".json"), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.size >= 1) {
      run(args(0))
    } else {
      val dateFormat = new SimpleDateFormat("yyyyMMdd")
      val currentDate = dateFormat.format(new Date())
      run(currentDate)
    }
  }

}
