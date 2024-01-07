package vlp.dep

/**
  * Possible different labels attached to a token.
  * 
  * phuonglh
  * 
  */
object Label extends Enumeration {
  val Id, Lemma, UniversalPartOfSpeech, PartOfSpeech, FeatureStructure, Head, DependencyLabel, SuperTag = Value 
}