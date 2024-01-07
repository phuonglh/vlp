package vlp.dep

/**
  * Created by phuonglh on 6/24/17.
  * 
  * A dependency graph for a sentence.
  */
case class Graph(sentence: Sentence) {
  
  /**
    * Gets the head -> dependent node id of the graph.
    * @return (token id -> its head id) map.
    */
  def heads: Map[String, String] = {
    var map = Map[String, String]()
    sentence.tokens.foreach(token => map += (token.id -> token.head))
    map
  }

  /**
    * Gets the (node -> dependency label) map of the graph 
    * @return (token id -> its dependency label) map.
    */
  def dependencies: Map[String, String] = {
    var map = Map[String, String]()
    sentence.tokens.foreach(token => map += (token.id -> token.dependencyLabel))
    map
  }

  /**
    * Checks whether this graph has an arc (u, v) or not. 
    * @param u a node id
    * @param v a node id
    * @return true or false
    */
  def hasArc(u: String, v: String): Boolean = {
    heads.exists(_ == (v, u))
  }

  /**
    * Checks whether there is an arc with a given dependency label going 
    * out from a token.
    * @param u token id
    * @param dependency dependency label 
    * @return true or false
    */
  def hasDependency(u: String, dependency: String): Boolean = {
    dependencies.exists(_ == (u, dependency))
  }

  override def toString: String = {
    val seq = dependencies.toSeq.sortBy(_._1).tail
    seq.map(pair => {
      val sb = new StringBuilder()
      val u = pair._1
      sb.append(pair._2)
      sb.append('(')
      sb.append(heads(u))
      sb.append('-')
      sb.append(sentence.token(heads(u)).word)
      sb.append(',')
      sb.append(u)
      sb.append('-')
      sb.append(sentence.token(u).word)
      sb.append(')')
      sb
    }).mkString("\n")
  }
}
