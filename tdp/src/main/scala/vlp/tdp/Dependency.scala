package vlp.tdp

/**
  * Created by phuonglh on 6/22/17.
  * 
  * A dependency is a triple of (head, dependent, label). 
  */
case class Dependency(head: String, dependent: String, label: String) {
  override def toString: String = {
    val sb = new StringBuilder
    sb.append(label)
    sb.append('(')
    sb.append(head)
    sb.append(',')
    sb.append(dependent)
    sb.append(')')
    sb.toString()
  }
}
