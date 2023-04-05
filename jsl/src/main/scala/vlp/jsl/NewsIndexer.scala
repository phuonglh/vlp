package vlp.jsl

import java.io.IOException
import java.net.{MalformedURLException, URL}
import java.text.SimpleDateFormat
import java.util.Date
import java.util.regex.Pattern

import de.l3s.boilerpipe.BoilerpipeProcessingException
import de.l3s.boilerpipe.extractors.ArticleExtractor
import org.apache.commons.io.IOUtils
import org.xml.sax.InputSource

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import scala.util.parsing.json._
import org.json4s.jackson.Serialization
import org.json4s._
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

/**
  * Extractor of health news content for a given day from some large online newswire. 
  * The news are sent to a Apache Kafka server.
  *
  */
object NewsIndexer {
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
          if (filter(url)) {
            urls += (site + url)
          }
        }
      }
    } catch {
      case e: IOException => e.printStackTrace()
      case e: Exception => 
        System.err.println(site + "/" + category)
        e.printStackTrace()
    } 
    urls.toSet
  }

  def runWithTimeout[T](timeout: Long)(f: => T): Option[T] = {
    try {
      Some(Await.result(Future(f), timeout.seconds))
    } catch {
      case e: TimeoutException => None
    }
  }

  def run(jsonSource: String): Unit = {
    System.setProperty("http.agent", "Chrome")
    System.setProperty("https.protocols", "TLSv1,TLSv1.1,TLSv1.2")

    import scala.collection.JavaConversions._
    val urls = mutable.Set[String]()
    // parse the sources.json file and run extraction
    val s = scala.io.Source.fromFile(jsonSource).getLines().toList.mkString("\n")
    val sites = JSON.parseFull(s).get.asInstanceOf[List[Map[String,Any]]]
    sites.foreach { site => 
      val source = site("site").asInstanceOf[String]
      val categories = site("category").asInstanceOf[List[String]]
      val pattern = site("pattern").asInstanceOf[String]
      val exclude = site("exclude").asInstanceOf[List[String]]
      categories.foreach { category => 
        urls ++= extractURLs(source, category, pattern, (s: String) => !exclude.exists(e => s.contains(e)))
      }
    }

    println(s"#(totalURLs) = ${urls.size}")

    val kafkaProducer = Kafka.createProducer(Kafka.SERVERS)
    val news = urls.par.map(url => {
      println(url)
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
      val dateFormat = new SimpleDateFormat("yyyyMMdd")
      val currentDate = dateFormat.format(new Date())
      Files.write(Paths.get(System.getProperty("user.dir"), "dat", currentDate + ".json"), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.size > 0) {
      run(args(0))
    } else run("dat/sources.json")
  }

}
