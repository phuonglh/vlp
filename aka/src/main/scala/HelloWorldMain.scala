import akka.actor.typed.scaladsl.Behaviors
import akka.actor.typed.{Behavior, ActorSystem}

object HelloWorldMain {
  final case class SayHello(name: String)

  def apply(): Behavior[SayHello] = {
    Behaviors.setup { context =>
      val greeter = context.spawn(HelloWorld(), "greeter")

      Behaviors.receiveMessage { message => 
        val replyTo = context.spawn(HelloWorldBot(max = 3), message.name)
        greeter ! HelloWorld.Greet(message.name, replyTo)
        Behaviors.same
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val system: ActorSystem[HelloWorldMain.SayHello] = ActorSystem(HelloWorldMain(), "hello")
    println("Starting the system...")
    system ! HelloWorldMain.SayHello("World")
    system ! HelloWorldMain.SayHello("Akka")
  }
}
