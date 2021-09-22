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
import java.util.regex.Pattern

/**
  * phuonglh, August 10, 2021.
  * 
  * Search evaluation on a test set.
  * 
  */

class Evaluator(host: String, port: Int, index: String = "qas") {

  val client: RestHighLevelClient = new RestHighLevelClient(RestClient.builder(new HttpHost(host, port, "http")))
  final val FIELD = "answer"
  final val PUNCTS = Pattern.compile("""[.,?!;:\"…/”“″=^▪•<>&«\])(\[\u0022\u200b\ufeff+-]+""")
  final val Q_WORDS = Pattern.compile("(hay|không|là|gì|nào|và|hoặc)+")


  /**
    * Process an input query to get a list of most important terms.
    *
    * @param query
    * @return a list of terms.
    */
  def processQuery(query: String): List[String] = {
    var matcher = PUNCTS.matcher(query)
    val x = matcher.replaceAll("")
    matcher = Q_WORDS.matcher(x)
    val y = matcher.replaceAll("").toLowerCase()
    return y.split("""\s+""").toList
  }

  def call(query: String, limit: Int): List[Q] = {
    val tokens = processQuery(query)
    if (tokens.length > 0)
      call(tokens, limit)
    else List.empty[Q]
  }

  def call(tokens: List[String], limit: Int): List[Q] = {
    val request = new SearchRequest("qas");
    val searchQuery = new SearchSourceBuilder()
    searchQuery.timeout(new TimeValue(60, TimeUnit.SECONDS))

    val bqb = QueryBuilders.boolQuery()
    for (token <- tokens) 
      bqb.must(QueryBuilders.termQuery(FIELD, token))

    println(bqb.toString) // for debugging
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
      val keywords = kws.substring(5, kws.length-1).split(",").toList
      Q(document.get("id").toString(), document.get("question").toString(), document.get("questionDetail").toString(), keywords)
    }
    return qs.toList
  }
  
  def eval(query: String, correctId: String) = {
    val ratedDocs = new ListBuffer[RatedDocument]()
    ratedDocs += (new RatedDocument(index, correctId, 1))
    val searchQuery = new SearchSourceBuilder()
    searchQuery.timeout(new TimeValue(60, TimeUnit.SECONDS))
    val bqb = QueryBuilders.boolQuery()
    val tokens = processQuery(query)
    println(tokens)
    for (token <- tokens) 
      bqb.must(QueryBuilders.termQuery(FIELD, token))
    println(bqb.toString)
    searchQuery.query(bqb)

    import scala.collection.JavaConversions._
    val ratedRequest = new RatedRequest(correctId, ratedDocs.toList, searchQuery)
    val ratedRequests = new ListBuffer[RatedRequest]()
    ratedRequests += ratedRequest

    val specification = new RankEvalSpec(ratedRequests, new MeanReciprocalRank(1, 5))
    val request = new RankEvalRequest(specification, Array(index))

    val response = client.rankEval(request, RequestOptions.DEFAULT)

    val score = response.getMetricScore()
    println(s"score = ${score}")

    val partialResults = response.getPartialResults()
    val rqQuality = partialResults.get(correctId)
    println(s"rqQuality = ${rqQuality.getId()}")
    println(s"rqQualityLevel = ${rqQuality.metricScore}")

    val hitsAndRatings = rqQuality.getHitsAndRatings()
    for (ratedSearchHit <- hitsAndRatings) {
      println("ratedSearchHit = " + ratedSearchHit.getSearchHit().getId() + ", is present? " + ratedSearchHit.getRating().isPresent())
    }
    val metricDetails = rqQuality.getMetricDetails()
    val detail = metricDetails.asInstanceOf[MeanReciprocalRank.Detail]
    println(detail.getFirstRelevantRank())
  }

  def eval(queries: List[String], correctIds: List[String]) = {
    import scala.collection.JavaConversions._
    val ratedDocs = new ListBuffer[RatedDocument]()
    val ratedRequests = new ListBuffer[RatedRequest]()

    for ((query, correctId) <- queries.zip(correctIds)) {
      ratedDocs += (new RatedDocument(index, correctId, 1))
      val searchQuery = new SearchSourceBuilder()
      val bqb = QueryBuilders.boolQuery()
      val tokens = processQuery(query)
      for (token <- tokens) 
        bqb.must(QueryBuilders.termQuery(FIELD, token))
      searchQuery.query(bqb)
      val ratedRequest = new RatedRequest(correctId, ratedDocs.toList, searchQuery)
      ratedRequests += ratedRequest
    }
    val specification = new RankEvalSpec(ratedRequests, new MeanReciprocalRank(1, 5))
    val request = new RankEvalRequest(specification, Array(index))

    val response = client.rankEval(request, RequestOptions.DEFAULT)
    val partialResults = response.getPartialResults()
    for (correctId <- correctIds) {
      val rqQuality = partialResults.get(correctId)
      println(s"rqQuality = ${rqQuality.getId()}")
      println(s"rqQualityLevel = ${rqQuality.metricScore}")

      val hitsAndRatings = rqQuality.getHitsAndRatings()
      for (ratedSearchHit <- hitsAndRatings) {
        println("ratedSearchHit = " + ratedSearchHit.getSearchHit().getId() + ", is present? " + ratedSearchHit.getRating().isPresent())
      }
      val metricDetails = rqQuality.getMetricDetails()
      val detail = metricDetails.asInstanceOf[MeanReciprocalRank.Detail]
      println(detail.getFirstRelevantRank())
    }
    val score = response.getMetricScore()
    println(s"average score = ${score}")
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

    val query = "ô nhiễm môi trường biển là gì?"

    // search for the query and print the top answers
    // val qs = evaluator.call(query, 5)
    // println("Number of answers = " + qs.size)
    // qs.foreach(println)

    // evaluate the answer of one request
    // val correctId = "7172"
    // evaluator.eval(query, correctId)

    // evaluate the answers of multiple requests
    // val queries = List("ô nhiễm môi trường biển là gì?", "quy trình cấp lại sổ đỏ")
    // val correctIds = List("7172", "7769")
    // evaluator.eval(queries, correctIds)

    // read test samples and evaluate the search system
    val samples = EvalCorpus.read(s"${System.getProperty("user.home")}" + "/vlp/dat/qas/result.json")
    // filter samples with negative results
    val negativeSamples = samples.filter(s => s.rankedIds.isEmpty)
    println(s"Number of negative samples = ${negativeSamples.size}." )

    // filter samples with positive results
    val positiveSamples = samples.filter(s => s.rankedIds.nonEmpty)
    println(s"Number of positive samples = ${positiveSamples.size}." )

    val queries = positiveSamples.map(s => s.question)
    val correctIds = positiveSamples.map(s => s.rankedIds.head)
    evaluator.eval(queries, correctIds)

    // val queries = negativeSamples.map(s => s.question)
    // val correctIds = (1 to queries.size).toList.map(_.toString())
    // evaluator.eval(queries, correctIds)

    println("Done.")
    evaluator.client.close()
  }
}
