package vlp.qas

import akka.actor.Actor

import scala.collection.mutable
import java.util.Properties

/**
  * Server-side implementation of the ranker.
  * 
  * phuonglh@gmail.com
  *
  */

case class Search(query: String)
case class Result(id: String, question: String)

class RankerActor(properties: Properties) extends Actor {
  val cache = new mutable.HashMap[String, List[Result]]()
  val host = properties.getProperty("host")
  val port = properties.getProperty("port").toInt
  val index = properties.getProperty("index")

  val evaluator = new Evaluator(host, port, index)
  
  override def receive: Receive = {
    case request: Search =>
      val result = if (!cache.keySet.contains(request.query)) {
        val tokens = evaluator.processQuery(request.query)
        var documents = evaluator.call(tokens, 5)
        if (documents.isEmpty) {
          println("Request too long. Reduce its by a half...")
          documents = evaluator.call(tokens.slice(0, tokens.length/2), 5)
        } 
        val rs = documents.map { d => 
          val q = d.question.take(80)
          val x = if (q.length < d.question.length) q + "..." else q
          Result(d.id, x)
        }
        cache += request.query -> rs
        rs        
      } else cache.get(request.query).get
      sender() ! result
    case "reset" =>
      cache.clear()
      sender() ! "reset"
  }
}