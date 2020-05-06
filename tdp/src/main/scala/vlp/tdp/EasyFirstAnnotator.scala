package vlp.tdp

import scala.collection.mutable.ListBuffer

/**
  * phuonglh, 8/5/17, 04:16
  * 
  * Easy-first rules for some token attachment, mostly concerning with neighboring tokens. 
  * Currently, we provide only some simple rules for Vietnamese dependency parsing.
  * 
  */

trait Rule extends Serializable {
  def f(a: Token, b: Token): String
}

class EasyFirstAnnotator(val language: Language.Value) {
  val VietnameseRules = List[Rule](new Tense, new Passive, new Case)

  /**
    * Annotates a sentence by using easy-first rules
    * @param sentence a sentence to annotate
    * @return a list of dependencies                
    */
  def annotate(sentence: Sentence): List[Dependency] = {
    val tokens = sentence.tokens
    val arcs = ListBuffer[Dependency]()
    language match {
      case Language.Vietnamese => {
        for (j <- 0 until (sentence.length - 1)) {
          for (r <- 0 until VietnameseRules.size) {
            val label = VietnameseRules(r).f(tokens(j), tokens(j+1))
            if (!label.isEmpty) {
              tokens(j).setHead(tokens(j+1).id)
              tokens(j).setDependencyLabel(label)
              arcs += Dependency(tokens(j+1).id, tokens(j).id, label)
            }
          }
        }
      }
      case Language.English => {}
    }
    arcs.toList
  }
}

class Tense extends Rule {
  val anchors = List[String]("đã", "sẽ", "sớm", "không", "chưa", "chẳng")
  override def f(a: Token, b: Token): String = {
    if (a.partOfSpeech == "R" && anchors.contains(a.word) && b.partOfSpeech == "V") "advmod" else ""
  }
}

class Passive extends Rule {
  val anchors = List[String]("bị", "được")
  override def f(a: Token, b: Token): String = {
    if (a.partOfSpeech == "V" && anchors.contains(a.word) && b.partOfSpeech == "V") "aux:pass" else ""
  }
}

class Case extends Rule {
  val anchors = List[String]("để")
  override def f(a: Token, b: Token): String = {
    if (a.partOfSpeech == "E" && anchors.contains(a.word) && b.partOfSpeech == "V") "case" else ""
  }
}
