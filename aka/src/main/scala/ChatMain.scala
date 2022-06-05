import akka.actor.typed.Behavior
import akka.NotUsed
import akka.actor.typed.scaladsl.Behaviors
import akka.actor.typed.Terminated
import akka.actor.typed.ActorSystem

object ChatMain {
  def apply(): Behavior[NotUsed] = // NotUsed since we do not need to send messages from the outside
    Behaviors.setup { context =>
      val chatRoom = context.spawn(ChatRoom(), "chatroom")
      val gabblerRef = context.spawn(Gabbler(), "gabbler")
      context.watch(gabblerRef)
      chatRoom ! ChatRoom.GetSession("ol' Gabbler", gabblerRef)

      Behaviors.receiveSignal {
        case (_, Terminated(_)) =>
          Behaviors.stopped
      }
    }

  def main(args: Array[String]): Unit = {
    ActorSystem(ChatMain(), "ChatRoomDemo")
  }
}
