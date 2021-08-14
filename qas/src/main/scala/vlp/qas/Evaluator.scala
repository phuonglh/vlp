package vlp.qas

import org.elasticsearch.search.builder.SearchSourceBuilder
import org.elasticsearch.index.query.QueryBuilders
import java.util.concurrent.TimeUnit
import org.elasticsearch.client.RequestOptions
import org.elasticsearch.client.RestHighLevelClient
import org.elasticsearch.client.RestClient
import org.apache.http.HttpHost
import org.elasticsearch.action.search.SearchRequest
import org.elasticsearch.common.unit.TimeValue
import java.util.Properties
import org.elasticsearch.index.rankeval.RatedDocument
import scala.collection.mutable.ListBuffer
import org.elasticsearch.index.rankeval.RatedRequest
import org.elasticsearch.index.rankeval.RankEvalSpec
import org.elasticsearch.index.rankeval.PrecisionAtK
import org.elasticsearch.index.rankeval.RankEvalRequest
import org.elasticsearch.index.rankeval.MeanReciprocalRank

/**
  * phuonglh, August 10, 2021.
  * 
  * 
  */

class Evaluator(host: String, port: Int, index: String = "qas") {

  val client: RestHighLevelClient = new RestHighLevelClient(RestClient.builder(new HttpHost(host, port, "http")))
  
  def call(index: String, query: String, limit: Int): List[Q] = {
    val request = new SearchRequest("qas");
    val searchQuery = new SearchSourceBuilder()
    searchQuery.timeout(new TimeValue(60, TimeUnit.SECONDS))
    val bqb = QueryBuilders.boolQuery()
    val tokens = query.split("\\s+")
    for (token <- tokens) 
      bqb.must(QueryBuilders.termQuery("question", token))
    searchQuery.query(bqb)

    searchQuery.size(limit)
    request.source(searchQuery)

    val response = client.search(request, RequestOptions.DEFAULT)

    val took = response.getTook()
    val time = took.getStringRep()
    println(time)

    val hits = response.getHits();
    val count = hits.getHits().length
    println(count)

    val hs = hits.getHits()
    val qs = hs.map { h => 
      val document = h.getSourceAsMap()
      val kws = document.get("questionKeywords").toString
      val keywords = kws.substring(6, kws.length-1).split(",").toList
      Q(document.get("id").toString(), document.get("question").toString(), document.get("questionDetail").toString(), keywords)
    }
    return qs.toList
  }
  
  def eval(query: String) = {
    val ratedDocs = new ListBuffer[RatedDocument]()
    ratedDocs += (new RatedDocument(index, "8251", 1))
    val searchQuery = new SearchSourceBuilder()
    searchQuery.timeout(new TimeValue(60, TimeUnit.SECONDS))
    val bqb = QueryBuilders.boolQuery()
    val tokens = query.split("\\s+")
    for (token <- tokens) 
      bqb.must(QueryBuilders.termQuery("question", token))

    import scala.collection.JavaConversions._
    val ratedRequest = new RatedRequest("rq1", ratedDocs.toList, searchQuery)
    val ratedRequests = new ListBuffer[RatedRequest]()
    ratedRequests += ratedRequest

    val specification = new RankEvalSpec(ratedRequests, new MeanReciprocalRank(1, 5))
    val request = new RankEvalRequest(specification, Array(index))

    val response = client.rankEval(request, RequestOptions.DEFAULT)

    val score = response.getMetricScore()
    println(s"score = ${score}")

    val partialResults = response.getPartialResults()
    val rq1Quality = partialResults.get("rq1")
    println(s"rq1Quality id = ${rq1Quality.getId()}")
    println(s"rq1QualityLevel = ${rq1Quality.metricScore}")

    val hitsAndRatings = rq1Quality.getHitsAndRatings()
    for (ratedSearchHit <- hitsAndRatings) {
      println("ratedSearchHit id = " + ratedSearchHit.getSearchHit().getId() + ", is present? " + ratedSearchHit.getRating().isPresent())
    }
    val metricDetails = rq1Quality.getMetricDetails()
    val detail = metricDetails.asInstanceOf[MeanReciprocalRank.Detail]
    println(detail.getFirstRelevantRank())
  }
}

object Evaluator {
  def main(args: Array[String]): Unit = {
    val properties = new Properties()
    properties.load(Evaluator.getClass().getClassLoader.getResourceAsStream("config.properties"))
    val host = properties.getProperty("host")
    val port = properties.getProperty("port").toInt
    val index = properties.getProperty("index")
   
    val evaluator = new Evaluator(host, port, index)
    // val query = "quy trình cấp lại sổ đỏ"
    // val query = "cấp lại sổ đỏ"
    val query = "chính sách bồi thường đất"
    val qs = evaluator.call(index, query, 5)
    qs.foreach(println)
    evaluator.eval(query)
    println("Done.")
    evaluator.client.close()
  }
}
