import akka.actor.typed.{ActorRef, ActorSystem, Behavior}
import akka.actor.typed.scaladsl.Behaviors

object HelloWorldBot {

  import HelloWorld._

  def apply(max: Int): Behavior[Greeted] = {
    bot(0, max)
  }

  private def bot(counter: Int, max: Int): Behavior[Greeted] = {
    Behaviors.receive { (context, message) =>
      val n = counter + 1
      context.log.info("Greeting {} for {}", n, message.whom)
      if (n == max) {
        Behaviors.stopped
      } else {
        message.from ! Greet(message.whom, context.self)
        bot(n, max)
      }
    }
  }
}
