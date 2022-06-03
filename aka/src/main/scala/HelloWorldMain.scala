import akka.actor.typed.scaladsl.Behaviors
import akka.actor.typed.{Behavior, ActorSystem}

object HelloWorldMain {
  final case class SayHello(name: String)
  import HelloWorld._

  def apply(): Behavior[SayHello] = {
    Behaviors.setup { context =>
      val greeter = context.spawn(HelloWorld(), "greeter")

      Behaviors.receiveMessage { message => 
        val replyTo = context.spawn(HelloWorldBot(max = 5), message.name)
        greeter ! Greet(message.name, replyTo)
        Behaviors.same
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val system: ActorSystem[SayHello] = ActorSystem(HelloWorldMain(), "helloActorSystem")
    system ! SayHello("World")
    system ! SayHello("Akka")
  }
}
