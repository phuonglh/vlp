package vlp.qas

import akka.actor.{ActorRef, ActorSystem}
import akka.util.Timeout

import scala.concurrent.ExecutionContext
import akka.pattern.ask

import akka.http.scaladsl.model._
import akka.http.scaladsl.model.HttpCharsets._
import akka.http.scaladsl.server.Route
import akka.http.scaladsl.server.Directives._

import org.json4s.Formats
import org.json4s.DefaultFormats
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import org.json4s.jackson.JsonMethods._
import de.heikoseeberger.akkahttpjackson.JacksonSupport

/**
  * Akka HTTP service for document ranker.
  * 
  * phuonglh@gmail.com
  */
trait RankerService extends JacksonSupport {
  val actor: ActorRef
  implicit def executionContext: ExecutionContext
  implicit def requestTimeout: Timeout
}

class RankerServiceAPI(
  system: ActorSystem,
  timeout: Timeout,
  val actor: ActorRef
) extends RankerService {
  implicit val requestTimeout = timeout // needed for ask
  implicit def executionContext = system.dispatcher // to call ask
  val routes = search ~ reset ~ test ~ query

  implicit val formats: Formats = DefaultFormats // json format

  def test: Route = pathPrefix("test") { ctx => 
    ctx.request.method match {
      case HttpMethods.GET => ctx.complete("Received GET")
      case HttpMethods.POST => ctx.complete("Received POST")
      case _ => ctx.complete("Received something else.")
    }
  }

  val directive = extract(ctx => ctx.request.uri.toString)

  def query: Route = pathPrefix("query") {
    directive { uri => 
      extractRequestEntity { entity =>
        complete(uri + s" with entity = ${entity}")
      }
    }
  }

  def search: Route = post {
    pathPrefix("search") {
      entity(as[Search]) { search =>
        // complete(search)
        onSuccess(actor ? search) { // or nameActor.ask(Search(query))
          case result: List[Any] =>
            complete(Serialization.write(result))
        }
      }
    }
  }

  def reset = path("reset") {
    get {
      onSuccess(actor ? "reset") {
        case _: String =>
          complete("Cache reset.")
      }
    }
  }
}


// curl -X POST -H "Content-Type: application/json; charset=utf-8" -d "{\"query\" : \"trang bị\"}" http://localhost:8085/search