package vlp.woz.act

import org.json4s._
import org.json4s.jackson.JsonMethods._
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets


/**
  * phuonglh, 2023
  * 
  */

case class Act(
  name: String,
  frames: List[(String, String)] // (slot, value)
)

case class Span(
  actName: String,
  slot: String,
  value: Any,
  start: Option[Int],
  end: Option[Int]
)

case class Turn(
  id: String,
  acts: Set[Act],
  spans: List[Span]
)

case class Dialog(
  id: String,
  turns: List[Turn]
)

object DialogActReader {

  def toInt(x: Any): Option[Int] = x match {
    case i: BigInt => Some(i.intValue())
    case _ => None
  }

  // /**
  //   * Reads a train/dev/test directory and return all *.json files.
  //   *
  //   * @param split
  //   */
  // def jsonPaths(split: String): List[String] = {
  //   val path = s"dat/woz/data/MultiWOZ_2.2/${split}"
  //   import scala.collection.JavaConverters._
  //   Files.list(Paths.get(path)).iterator().asScala.map(_.toString).filter(_.endsWith(".json")).toList
  // }

  /**
    * Reads the dialog act file and return a sequence of dialogs.
    *
    * @param path
    */
  def readDialogs(path: String): Seq[Dialog] = {
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
      val turns = turnIds.map { id =>
        val elements = turnMap(id).asInstanceOf[Map[String, Any]]
        // get acts
        val dialogAct = elements("dialog_act").asInstanceOf[Map[String, List[List[String]]]]
        val acts = dialogAct.keySet.map { name => 
          val frames = dialogAct(name).map(list => (list(0), list(1)))
          Act(name, frames)
        }
        // get spans 
        val spanInfo = elements("span_info").asInstanceOf[List[List[Any]]]
        val spans = spanInfo.map { list => 
          Span(list(0).asInstanceOf[String], list(1).asInstanceOf[String], list(2), toInt(list(3)), toInt(list(4)))
        }
        Turn(id, acts, spans)
      }
      Dialog(id, turns)
    }
  }
  /**
    * Extracts a sequence of triples (dialogId, turnId, [actName1, actName2]).
    * Normally, a turn has one act or two acts. 
    *
    * @param ds
    * @return a sequence of triples (dialogId, turnId, actNames)
    */
  def extractActNames(ds: Seq[Dialog]): Seq[(String, String, List[String])] = {
    ds.toList.flatMap(d => d.turns.map(t => (d.id, t.id, t.acts.toList.map(_.name))))
  }

  def readAll(): Seq[(String, String, List[String])] = {
    val ds = readDialogs("dat/woz/data/MultiWOZ_2.2/dialog_acts.json")
    println(s"Number of dialogs = ${ds.size}")
    extractActNames(ds)
  }

  def main(args: Array[String]): Unit = {
    val as = readAll()
    println(s"Number of turns = ${as.size}")
    as.toList.take(20).foreach(println)
  }
}
