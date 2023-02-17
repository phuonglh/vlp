package vlp.woz.act

import scala.util.parsing.json._
import org.json4s._
import org.json4s.jackson.Serialization
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets
import scala.collection.mutable.ListBuffer


case class Act(
  name: String,
  frames: List[(String, String)] // (slot, value)
)

case class Turn(
  id: String,
  act: Set[Act]
)

case class Dialog(
  id: String,
  turns: List[Turn]
)

object DialogActFilter {
  def main(args: Array[String]): Unit = {
    implicit val formats = DefaultFormats
    val path = "/Users/phuonglh/vlp/woz/dat/woz/003.json"
    val content = scala.io.Source.fromFile(path).getLines().toList.mkString(" ")
    // unmarshal to a map of dialogs 
    val dialogs = JSON.parseFull(content).get.asInstanceOf[Map[String,Any]]
    // extract dialog ids and turns
    val ds = dialogs.keySet.map { id =>
      val turnMap = dialogs(id).asInstanceOf[Map[String, Any]]
      // get sorted turnIds
      val turnIds = turnMap.keySet.toList.sorted
      // get corresponding acts
      val turns = turnIds.map { id =>
        val elements = turnMap(id).asInstanceOf[Map[String, Any]]
        val dialogAct = elements.get("dialog_act").asInstanceOf[Map[String, List[List[String]]]]
        val acts = dialogAct.keySet.map { name => 
          val frames = dialogAct(name).map(list => (list(0), list(1)))
          Act(name, frames)
        }
        Turn(id, acts)
      }
      Dialog(id, turns)
    }
    ds.foreach(println)
  }
}
