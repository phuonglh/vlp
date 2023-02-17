package vlp.woz.act

import org.json4s._
import org.json4s.jackson.JsonMethods._
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets


case class Act(
  name: String,
  frames: List[(String, String)] // (slot, value)
)

case class Turn(
  id: String,
  acts: Set[Act]
)

case class Dialog(
  id: String,
  turns: List[Turn]
)

object DialogActReader {
  /**
    * Reads the dialog act file and return a sequence of dialogs.
    *
    * @param path
    */
  def read(path: String): Seq[Dialog] = {
    implicit val formats = DefaultFormats    
    val content = scala.io.Source.fromFile(path).getLines().toList.mkString("\n")
    val ds = parse(content)
    // pretty(render(ds))    
    // get a map of dialogs using the
    val dialogs = ds.values.asInstanceOf[Map[String,Any]]
    // extract dialogs
    dialogs.keySet.toSeq.map { id =>
      val turnMap = dialogs(id).asInstanceOf[Map[String, Any]]
      // get sorted turnIds (need to convert string id to int id in the sort function)
      val turnIds = turnMap.keySet.toList.sortBy(_.toInt)
      // get corresponding acts
      val turns = turnIds.map { id =>
        val elements = turnMap(id).asInstanceOf[Map[String, Any]]
        val dialogAct = elements("dialog_act").asInstanceOf[Map[String, List[List[String]]]]
        val acts = dialogAct.keySet.map { name => 
          val frames = dialogAct(name).map(list => (list(0), list(1)))
          Act(name, frames)
        }
        Turn(id, acts)
      }
      Dialog(id, turns)
    }
  }
  /**
    * Extracts a sequence of triples (dialogId, turnId, actName)
    *
    * @param ds
    */
  def readActNames(ds: Seq[Dialog]): Seq[(String, String, String)] = {
    ds.toList.flatMap(d => d.turns.map(t => (d.id, t.id, if (t.acts.size > 0)t.acts.toSeq.head.name else "")))
  }

  def main(args: Array[String]): Unit = {
    val path = if (args.size == 0) "dat/woz/003.json" else args(0)
    val ds = read(path)
    println(s"Number of dialogs = ${ds.size}")
    val as = readActNames(ds)
    println(s"Number of turns = ${as.size}")
    as.toList.take(10).foreach(println)
  }
}
