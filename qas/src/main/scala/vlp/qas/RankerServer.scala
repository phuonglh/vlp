package vlp.qas


import akka.actor.{ActorSystem, Props}
import akka.event.Logging
import akka.http.scaladsl.Http
import akka.stream.ActorMaterializer
import akka.util.Timeout
import com.typesafe.config.{Config, ConfigFactory}

import scala.concurrent.Future
import scala.io.StdIn

/**
  *
  * Ranker server which runs a ranker service.
  * 
  * phuonglh@gmail.com
  */
object RankerServer extends App with RequestTimeout {
  val config = ConfigFactory.load()
  val host = config.getString("http.host")
  val port = config.getString("http.port")

  implicit val system = ActorSystem()
  implicit val ec = system.dispatcher

  val actor = system.actorOf(Props(new RankerActor), "rankerActor")
  val api = new RankerServiceAPI(system, requestTimeout(config), actor).routes
  implicit val materializer = ActorMaterializer()
  val bindingFuture: Future[Http.ServerBinding] = Http().bindAndHandle(api, host, Integer.parseInt(port))
  val log = Logging(system.eventStream, "rankerService")
  bindingFuture.map { serverBinding =>
    log.info(s"Bound to ${serverBinding.localAddress}")
  }.onFailure {
    case exception: Exception =>
      log.error(exception, "Failed to bind to {}:{}!", host, port)
      system.terminate()
  }

  // let it run until user presses Return at that time, unbind the port and shutdown  
  // StdIn.readLine()
  // bindingFuture.flatMap(_.unbind()).onComplete(_ => system.terminate())
}

/**
  * Reads request timeout from the configuration of the application.
  */
trait RequestTimeout {
  import scala.concurrent.duration._
  def requestTimeout(config: Config): Timeout = {
    val t = config.getString("akka.http.server.request-timeout")
    val d = Duration(t)
    FiniteDuration(d.length, d.unit)
  }
}