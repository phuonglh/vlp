package vlp.ner

import scala.collection.immutable.Map
import scala.collection.mutable.ListBuffer

/**
 * @author Phuong LE-HONG
 * <p>
 * Sep 6, 2016, 4:53:44 PM
 * <p>
 * A sentence is a list of tokens.
 * 
 */
case class Sentence(tokens: ListBuffer[Token]) {
  def length: Int = tokens.length
  def slice(startIndex: Int, endIndex: Int): Sentence = {
    Sentence(tokens.slice(startIndex, endIndex))
  }
}

/**
  * @author Phuong LE-HONG
  * <p>
  * Sep 6, 2016, 4:44:56 PM
  * <p>
  * A token is specified by a surface word and annotations, which are coded using
  * a map of (key, value) pairs. The annotation map includes part-of-speech, chunk
  * and named entity tags and their corresponding values.
  *
  */
case class Token(word: String, annotation: Map[Label.Value, String]) extends Serializable {
  override def toString(): String = {
    val s = new StringBuilder()
    s.append("Token(")
    s.append(word)
    s.append(",[")
    if (!annotation.keys.isEmpty) {
      val a = new StringBuilder()
      annotation.keys.foreach {
        k => {
          a.append(k.toString)
          a.append("=")
          a.append(annotation(k))
          a.append(' ')
        }
      }
      s.append(a.toString.trim)
    }
    s.append("])")
    s.toString()
  }

  def partOfSpeech: String = annotation.getOrElse(Label.PartOfSpeech, None.toString)
  def chunk: String = annotation.getOrElse(Label.Chunk, None.toString)
  def namedEntity: String = annotation.getOrElse(Label.NamedEntity, None.toString)
  def regexpType: String = annotation.getOrElse(Label.RegexpType, None.toString)
}
