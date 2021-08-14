package vlp.qas

import scala.util.parsing.json._
import org.apache.spark.rdd.RDD
import org.json4s.jackson.Serialization
import org.json4s._
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets
import scala.collection.mutable.ArrayBuffer

import org.apache.http.HttpHost
import org.elasticsearch.action.ActionListener
import org.elasticsearch.action.bulk.BulkRequest
import org.elasticsearch.action.bulk.BulkResponse
import org.elasticsearch.action.index.IndexRequest
import org.elasticsearch.action.index.IndexResponse
import org.elasticsearch.action.search.SearchRequest
import org.elasticsearch.action.search.SearchResponse
import org.elasticsearch.client.RequestOptions
import org.elasticsearch.client.RestClient
import org.elasticsearch.client.RestHighLevelClient
import org.elasticsearch.common.unit.TimeValue
import org.elasticsearch.common.xcontent.XContentBuilder
import org.elasticsearch.index.query.QueryBuilders
import org.elasticsearch.search.SearchHit
import org.elasticsearch.search.builder.SearchSourceBuilder
import org.elasticsearch.search.sort.SortOrder
import org.elasticsearch.common.xcontent.XContentFactory.jsonBuilder
import java.util.regex.Pattern

/** 
* Read q/a samples and index into a ES server. 
* (C) phuonglh@gmail.com, August
* 8, 2021.
*/
object Indexer {
  val HOST = "localhost"
  val client: RestHighLevelClient = new RestHighLevelClient(RestClient.builder(new HttpHost(HOST, 9200, "http")));

    final val HTML_PATTERN = Pattern.compile("<.+?>")

    def removeTags(text: String): String = {
        val matcher = HTML_PATTERN.matcher(text)
        return matcher.replaceAll("")
    }

  /**
    * Index a list of QA samples. Each element has an ID that matches the given ID to facilitate 
    * ranking evaluation.
    *
    * @param samples
    */
  def index(samples: List[QA], indexName: String = "qas") {
    val request = new BulkRequest()
    samples.foreach { sample => 
      val builder: XContentBuilder = jsonBuilder().startObject()
      val id = sample.id.get.toString().toFloat.toInt
      val keywords = sample.questionKeywords.get.toString
      builder.field("id", id)
        .field("question", sample.question)
        .field("questionDetail", sample.questionDetail)
        .field("answer", sample.answer)
        .field("questionKeywords", keywords)
        .field("questionType", sample.questionType)
        .field("area", sample.area)
        .field("source", sample.source)
      builder.endObject()
      request.add(new IndexRequest(indexName).id(String.valueOf(id)).source(builder))
    }
    client.bulkAsync(request, RequestOptions.DEFAULT, new ActionListener[BulkResponse]() {
      def onResponse(bulkItemResponses: BulkResponse) {
        println("Success in indexing " + samples.length + " samples.")
      }
      def onFailure(e: Exception) {
        e.printStackTrace();
        println("Failure in indexing " + samples.length + " samples.")
      }
    });
  }

  def main(args: Array[String]): Unit = {
    implicit val formats = Serialization.formats(NoTypeHints)
    import scala.collection.JavaConversions._
    val path = s"${System.getProperty("user.home")}" + "/vlp/dat/qas/all.jsonl"
    val qas = ArrayBuffer[QA]()
    val elements = scala.io.Source.fromFile(path)("UTF-8").getLines().toList.foreach { line => 
      val obj = JSON.parseFull(line).get.asInstanceOf[Map[String, Any]]
      if (obj.get("question") != None && obj.get("idSqlManage") != None) {
          val qa = QA(obj.get("idSqlManage"),
              obj.get("question").get.toString().replaceAll("""[\s\r]+""", " "),
              removeTags(obj.get("questionDetail").get.toString()).replaceAll("""[\s\r]+""", " "),
              obj.get("answer").get.toString().replaceAll("""[\s\r]+""", " "),
              obj.get("questionKeyword"),
              obj.getOrElse("loaiCauHoi", "NA").toString(), 
              obj.getOrElse("linhVuc", "NA").toString, 
              obj.getOrElse("nguon", "NA").toString())
          qas.append(qa)
      }
    }
    println(qas.length)
    import scala.collection.JavaConversions._
    val samples = qas.toList
    index(samples, "qas")
    Thread.sleep(5000)
    client.close()
    println("Done.")
  }
}
