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

class RankerActor(properties: Properties) extends Actor {
  val cache = new mutable.HashMap[String, String]()
  val host = properties.getProperty("host")
  val port = properties.getProperty("port").toInt
  val index = properties.getProperty("index")

  val evaluator = new Evaluator(host, port, index)
  
  override def receive: Receive = {
    case request: Search =>
      val result = if (!cache.keySet.contains(request.query)) {
        val documents = evaluator.call(request.query, 5)
        cache += request.query -> documents.toString
        documents
      } else cache.get(request.query).get
      // return result
      sender() ! result
    case "reset" =>
      cache.clear()
      sender() ! "reset"
  }
}