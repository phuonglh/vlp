package vlp.qas

import org.json4s.DefaultFormats
import org.json4s._
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.Serialization

import scala.collection.mutable.ListBuffer
import scala.io.Source

import scalaj.http._

/**
 * An utility to query documents from an ElasticSearch server.
 * 
 * phuonglh@gmail.com
 * 
 */
object Ranker {

    val summaryApiPrefix = "http://localhost:9200/qas/_search?q=id:"

    /**
      * Submits a query to the API and get a list of documents.
      *
      * @param keyword
      * @return a list of documents.
      */
    def query(id: String): Unit = {
        val summaryApi = summaryApiPrefix + id
        val response: HttpResponse[String] = Http(summaryApi).asString
        implicit val formats = DefaultFormats
        val jsArray = parse(response.body)
        // val result = ListBuffer[Article]()
        // for (articles <- jsArray.children) {
        //     for (article <- articles.children)
        //         result += article.extract[Article]
        // }
        // result.toList
        println(jsArray)
    }

    def main(args: Array[String]): Unit = {
        query("100")
    }
}
