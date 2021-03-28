package vlp.tmi

import org.json4s.DefaultFormats
import org.json4s._
import org.json4s.jackson.JsonMethods.parse

import scala.collection.mutable.ListBuffer
import scala.io.Source

import scalaj.http._

/**
 * An utility to query news from the GDELT Project through its Summary API.
 * 
 * phuonglh@gmail.com, March 2021.
 * 
 */
object GDELT {
    val summaryApiPrefix = "http://api.gdeltproject.org/api/v2/doc/doc?query="
    val summaryApiSuffix = "%20sourcecountry:vietnam&mode=artlist&timespan=1day&sort=datedesc&format=json"

    /**
      * Submits a query to the GDELT API and get a list of articles.
      *
      * @param keyword
      * @return a list of articles.
      */
    def query(keyword: String): List[Article] = {
        val summaryApi = summaryApiPrefix + keyword + summaryApiSuffix
        val response: HttpResponse[String] = Http(summaryApi).asString
        implicit val formats = DefaultFormats
        val jsArray = parse(response.body)
        val result = ListBuffer[Article]()
        for (articles <- jsArray.children) {
            for (article <- articles.children)
                result += article.extract[Article]
        }
        result.toList
    }

    /**
      * Extracts the contents from a list of articles.
      *
      * @param articles
      */
    def extract(articles: List[Article]): List[Document] = {
        articles.map(a => Document(a.url, NewsIndexer.extract(a.url)))
    }
    

    def main(args: Array[String]): Unit = {
        val articles = query("environment")
        println(articles.size)
        articles.foreach(println)
        val documents = extract(articles)
        documents.foreach(println)
    }
}
