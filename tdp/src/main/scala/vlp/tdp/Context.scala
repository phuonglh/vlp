package vlp.tdp

/**
  * Created by phuonglh on 6/22/17.
  * 
  * A parsing context contains space-separated feature strings and a transition label.
  * 
  */
case class Context(id: Int, bof: String, transition: String) {
  override def toString: String = {
    val sb = new StringBuilder()
    sb.append('(')
    sb.append(id)
    sb.append(',')
    sb.append(bof)
    sb.append(',')
    sb.append(transition)
    sb.append(')')
    sb.toString()
  }
}

/**
  * Extended parsing context
  * @param id
  * @param bof
  * @param transition
  * @param s word vector of the top element on the parsing stack
  * @param q word vector of the front element of the parsing queue
  */
case class ExtendedContext(id: Int, bof: String, transition: String, s: Vector[Double], q: Vector[Double])