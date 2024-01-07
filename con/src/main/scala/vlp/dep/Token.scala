package vlp.dep

import scala.collection.mutable

/**
  * A token contains a word and an annotation map of different (label, value) pairs.
  *
  * @param word
  * @param annotation
  */
case class Token(word: String, annotation: mutable.Map[Label.Value, String]) extends Serializable {
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
  
  def id: String = annotation.getOrElse(Label.Id, None.toString)
  def lemma: String = annotation.getOrElse(Label.Lemma, None.toString)
  def universalPartOfSpeech: String = annotation.getOrElse(Label.UniversalPartOfSpeech, None.toString)
  def partOfSpeech: String = annotation.getOrElse(Label.PartOfSpeech, None.toString)
  def featureStructure: String = annotation.getOrElse(Label.FeatureStructure, None.toString)
  def head: String = annotation.getOrElse(Label.Head, None.toString)
  def dependencyLabel: String = annotation.getOrElse(Label.DependencyLabel, None.toString)
  def superTag: String = annotation.getOrElse(Label.SuperTag, None.toString)

  /**
    * Updates the head information for this token.
    * @param head
    */
  def setHead(head: String): Unit = {
    annotation += (Label.Head -> head)
  }

  /**
    * Updates the dependency label for this token.
    * @param label
    */
  def setDependencyLabel(label: String): Unit = {
    annotation += (Label.DependencyLabel -> label)
  }
}
