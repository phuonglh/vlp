package vlp.tdp

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source

/**
  * Created by phuonglh on 6/24/17.
  */
object GraphReader {
  val root = Token("ROOT", mutable.Map[Label.Value, String](Label.Id -> "0"))
  
  /**
    * Reads a dependency graph corpus using Universal Dependency format.
    * @param path
    * @return a list of dependency graphs.
    */
  def read(path: String): List[Graph] = {
    val lines = (Source.fromFile(path, "UTF-8").getLines().filterNot(line => line.startsWith("#")) ++ List("")).toArray
    val graphs = new ListBuffer[Graph]()
    val indices = lines.zipWithIndex.filter(p => p._1.trim.isEmpty).map(p => p._2)
    var u = 0
    var v = 0
    for (i <- (0 until indices.length)) {
      v = indices(i)
      if (v > u) { // don't treat two consecutive empty lines
        val s = lines.slice(u, v)
        val tokens = s.map(line => {
          val parts = line.trim.split("\\t+")
          Token(parts(1).replaceAll("\\s+", "_"), mutable.Map(Label.Id -> parts(0),
            Label.Lemma -> parts(2),
            Label.UniversalPartOfSpeech -> parts(3),
            Label.PartOfSpeech -> parts(4), 
            Label.FeatureStructure -> parts(5), 
            Label.Head -> parts(6),
            Label.DependencyLabel -> parts(7),
            Label.SuperTag -> parts(parts.length - 1)))
        })
        val x = root +: tokens
        graphs.append(Graph(Sentence(x.toList.to[ListBuffer])))
      }
      u = v + 1
    }
    graphs.toList
  }
  
}
