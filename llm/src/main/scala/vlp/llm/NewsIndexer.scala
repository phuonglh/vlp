package vlp.llm

import java.io.IOException
import java.net.{MalformedURLException, URL}
import java.text.SimpleDateFormat
import java.util.Date
import java.util.regex.Pattern

import org.apache.commons.io.IOUtils
import org.xml.sax.InputSource
import de.l3s.boilerpipe.BoilerpipeProcessingException
import de.l3s.boilerpipe.extractors.ArticleExtractor

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.DefaultFormats._

import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets

import scala.concurrent.Await
import scala.concurrent.Future
import scala.concurrent.ExecutionContext
import java.util.concurrent.TimeoutException
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import org.apache.kafka.clients.producer.ProducerRecord

case class Page(url: String, content: String, date: Date)
case class Site(site: String, category: List[String], pattern: String, exclude: List[String])
case class Sites(sites: List[Site])

/**
  * Extractor of news content some large online newswire agencies. The data sources are specified 
  * in the `dat/sources.json` file. Extracted news are sent to a Apache Kafka server as well as 
  * saved to a JSON file in the 'dat/' directory.
  * <p/>
  * (C) phuonglh@gmail.com, April 2023.
  */
object NewsIndexer {
  val specialUnicodePattern = Pattern.compile("""[\u0000]+""")
  val useKafka: Boolean = false

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
      case e: MalformedURLException => System.err.println(e.getMessage); ""
      case e: BoilerpipeProcessingException => System.err.println(e.getMessage); ""
      case e: IOException => System.err.println(e.getMessage); ""
      case _: Exception => System.err.println("Other exception"); ""
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
          urls += (site + url)
        }
      }
    } catch {
      case e: IOException => e.printStackTrace()
      case e: Exception => 
        System.err.println(site + "/" + category)
        e.printStackTrace()
    } 
    urls.toSet.filter(filter(_))
  }

  def runWithTimeout[T](timeout: Long)(f: => T): Option[T] = {
    try {
      Some(Await.result(Future(f), timeout.seconds))
    } catch {
      case _: TimeoutException =>
        println("Timeout exception!")
        None
    }
  }

  def run(jsonSource: String): Unit = {
    System.setProperty("http.agent", "Chrome")
    System.setProperty("https.protocols", "TLSv1,TLSv1.1,TLSv1.2")

    val urls = mutable.Set[String]()
    val s = scala.io.Source.fromFile(jsonSource).getLines().toList.mkString("\n")
    implicit val formats: Formats = Serialization.formats(ShortTypeHints(List(classOf[Site])))
    val json = Serialization.read[Sites](s)
    json.sites.foreach { site => 
      site.category.foreach { category => 
        urls ++= extractURLs(site.site, category, site.pattern, (s: String) => !site.exclude.exists(e => s.contains(e)))
      }
    }
    // extract articles in parallel
    val news = if (useKafka) {
      val kafkaProducer = Kafka.createProducer(Kafka.SERVERS)
      val ns = urls.par.map(url => {
        println(url)
        val content = runWithTimeout(4000)(extract(url)).get
        if (content.size >= 500 && !content.contains("div") && !content.contains("class=") && !content.contains("script") && !specialUnicodePattern.matcher(content).find()) {
          kafkaProducer.send(new ProducerRecord[String, String](Kafka.GROUP_ID, url, content))
          Page(url, content, new Date())
        } else {
          Page(url, "", new Date())
        }
      }).toList.filter(_.content.nonEmpty)
      kafkaProducer.close()
      ns
    } else {
      urls.par.map(url => {
        println(url)
        val content = runWithTimeout(4000)(extract(url)).get
        if (content.size >= 500 && !content.contains("div") && !content.contains("class=") && !content.contains("script") && !specialUnicodePattern.matcher(content).find()) {
          Page(url, content, new Date())
        } else {
          Page(url, "", new Date())
        }
      }).toList.filter(_.content.nonEmpty)
    }

    if (news.nonEmpty) {
      implicit val formats: Formats = Serialization.formats(NoTypeHints)
      val content = Serialization.writePretty(news)
      val dateFormat = new SimpleDateFormat("yyyyMMdd")
      val currentDate = dateFormat.format(new Date())
      Files.write(Paths.get(System.getProperty("user.dir"), "dat/idx", currentDate + ".json"), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    }
    println(s"#(totalURLs) = ${urls.size}")
  }

  def main(args: Array[String]): Unit = {
    if (args.size > 0) {
      run(args(0))
    } else run("dat/sources.json")
  }

}
