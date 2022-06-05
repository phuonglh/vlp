import akka.actor.typed.Behavior
import akka.actor.typed.scaladsl.Behaviors


object Gabbler {
  import ChatRoom._

  def apply(): Behavior[SessionEvent] =
    Behaviors.setup { context =>
      Behaviors.receiveMessage {
        case SessionGranted(handle) =>
          handle ! PostMessage("Hello World!")
          handle ! PostMessage("Bonjour Monde!")
          handle ! PostMessage("Xin chao!")
          handle ! PostMessage("bye")
          Behaviors.same
        case MessagePosted(screenName, message) =>
          context.log.info(s"message has been posted by '${screenName}': ${message}")
          if (message == "bye")
            Behaviors.stopped
          else Behaviors.same
        case SessionDenied(reason) => 
          context.log.info(s"session denied because of ${reason}")
          Behaviors.stopped
      }
    }
}
