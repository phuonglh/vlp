package vlp.ner

import scala.collection.immutable.Map
import scala.collection.mutable.ListBuffer
import vlp.tok.WordShape

/**
 * @author Phuong LE-HONG
 * <p>
 * Sep 6, 2016, 4:54:26 PM
 * <p>
 * A label context contains a word, a tag and a string representing a bag of 
 * feature strings.
 */
case class LabeledContext(word: String, tag: String, bof: String) extends Serializable 

/**
 * @author Phuong LE-HONG
 * <p>
 * Sep 5, 2016, 12:30:40 PM
 * <p>
 * A featurizer which builds features for NER or POS.
 * 
 */
object Featurizer extends Serializable {
  
  /**
   * Extracts a labeled context at a position of a sentence.
   * @param sentence
   * @param position
   * @param extendedFeatureSet
   * @return a labeled context object.
   */
  def extract(sentence: Sentence, position: Int, extendedFeatureSet: Set[String]): LabeledContext = {
    def basicFeatures(): String = {
      val features = new StringBuilder()
      val currentToken = sentence.tokens(position)
      // current word
      features.append("w(0)=")
      features.append(currentToken.word)
      features.append(' ')
      if (extendedFeatureSet.contains("pos")) {
        // current part-of-speech
        features.append("p(0)=")
        features.append(currentToken.partOfSpeech)
        features.append(' ')
      }
      if (extendedFeatureSet.contains("chunk")) {
        // current chunk
        features.append("c(0)=")
        features.append(currentToken.chunk)
        features.append(' ')
      }
      // previous tag or "BOS" sentinel (begin of sentence)
      features.append("t(-1)=")
      features.append(if (position > 0) sentence.tokens(position - 1).namedEntity else "BOS")
      features.append(' ')
      // previous of previous tag or "BOS"
      features.append("t(-2)=")
      features.append(if (position > 1) sentence.tokens(position - 2).namedEntity else "BOS")
      features.append(' ')
      // return entire feature string
      features.toString().trim()
    }
    
    def wordShapeFeature(): String = {
      val currentWord = sentence.tokens(position).word
      val shape = WordShape.shape(currentWord) 
      if (!shape.isEmpty) "shape=" + shape else ""
    }
    
    def jointFeatures(): String = {
      val features = new StringBuilder()
      // w(-1)
      val prevWord = if (position > 0) sentence.tokens(position - 1).word else "BOS"
      features.append("w(-1)=")
      features.append(prevWord)
      features.append(' ')
      // w(0)+w(-1)
      val currentToken = sentence.tokens(position)
      features.append("w(0)+w(-1)=")
      features.append(currentToken.word)
      features.append('+')
      features.append(prevWord)
      features.append(' ')
      // w(+1)
      val nextWord = if (position < sentence.length-1) sentence.tokens(position + 1).word else "EOS"
      features.append("w(+1)=")
      features.append(nextWord)
      features.append(' ')
      // w(0)+w(+1)
      features.append("w(0)+w(+1)=")
      features.append(currentToken.word)
      features.append('+')
      features.append(nextWord)
      features.append(' ')
      if (extendedFeatureSet.contains("pos")) {
        // p(-1)
        val prevPoS = if (position > 0) sentence.tokens(position - 1).partOfSpeech else "BOS"
        features.append("p(-1)=")
        features.append(prevPoS)
        features.append(' ')
        // p(0)+p(-1)
        features.append("p(0)+p(-1)=")
        features.append(currentToken.partOfSpeech)
        features.append('+')
        features.append(prevPoS)
        features.append(' ')
        // p(+1)
        val nextPoS = if (position < sentence.length-1) sentence.tokens(position + 1).partOfSpeech else "EOS"
        features.append("p(+1)=")
        features.append(nextPoS)
        features.append(' ')
        // p(0)+p(+1)
        features.append("p(0)+p(+1)=")
        features.append(currentToken.partOfSpeech)
        features.append('+')
        features.append(nextPoS)
        features.append(' ')
        // p(-1)+p(+1)
        features.append("p(-1)+p(+1)=")
        features.append(prevPoS)
        features.append('+')
        features.append(nextPoS)
        features.append(' ')
        // w(0)+t(-1)
        features.append("w(0)+t(-1)=")
        features.append(currentToken.word)
        features.append('+')
        features.append(if (position > 0) sentence.tokens(position - 1).namedEntity else "BOS")
        features.append(' ')
      }
      // return the entire feature string
      features.toString().trim()
    }
    
    def regexpFeatures() = {
      val features = new StringBuilder()
      val currentToken = sentence.tokens(position)
      // r(0)
      val currentReg = currentToken.annotation.getOrElse(Label.RegexpType, "NA")
      features.append("r(0)=")
      features.append(currentReg)
      features.append(' ')
      // r(-1)
      val prevReg = if (position > 0) sentence.tokens(position-1).annotation.getOrElse(Label.RegexpType, "NA") else "BOS"
      features.append("r(-1)=")
      features.append(prevReg)
      features.append(' ')
      // r(0)+r(-1)
      features.append("r(0)+r(-1)=")
      features.append(currentReg)
      features.append('+')
      features.append(prevReg)
      features.append(' ')
      // r(+1)
      val nextReg = if (position < sentence.length-1) sentence.tokens(position+1).annotation.getOrElse(Label.RegexpType, "NA") else "EOS"
      features.append("r(+1)=")
      features.append(nextReg)
      features.append(' ')
      // r(0)+r(+1)
      features.append("r(0)+r(+1)=")
      features.append(currentReg)
      features.append('+')
      features.append(nextReg)
      features.append(' ')
      // w(0)+r(0)
      features.append("w(0)+r(0)=")
      features.append(currentToken.word)
      features.append('+')
      features.append(currentReg)
      features.append(' ')
      // w(0)+r(-1)
      features.append("w(0)+r(-1)=")
      features.append(currentToken.word)
      features.append('+')
      features.append(prevReg)
      features.append(' ')
      // w(0)+r(+1)
      features.append("w(0)+r(+1)=")
      features.append(currentToken.word)
      features.append('+')
      features.append(nextReg)
      features.append(' ')
      
      // t(0)+r(0)
      features.append("p(0)+r(0)=")
      features.append(currentToken.partOfSpeech)
      features.append('+')
      features.append(currentReg)
      features.append(' ')
      // t(0)+r(-1)
      features.append("p(0)+r(-1)=")
      features.append(currentToken.partOfSpeech)
      features.append('+')
      features.append(prevReg)
      features.append(' ')
      // t(0)+r(+1)
      features.append("p(0)+r(+1)=")
      features.append(currentToken.partOfSpeech)
      features.append('+')
      features.append(nextReg)
      features.append(' ')
      
      features.toString().trim()
    }
    
    // build a string containing all space-separated feature strings 
    val features = new StringBuilder() 
    features.append(basicFeatures())
    features.append(' ')
    features.append(wordShapeFeature())
    features.append(' ')
    features.append(jointFeatures())
    features.append(' ')
    if (extendedFeatureSet.contains("regexp")) {
      // first, run a wordsRegexp on this sentence to annotate regexp type for each token 
      WordRegexp.annotate(sentence)
      features.append(regexpFeatures()) 
    }
    // create a labeled context 
    val currentToken = sentence.tokens(position)
    new LabeledContext(currentToken.word, currentToken.namedEntity, features.toString)
  }

  /**
   * Extracts a list of labeled contexts from a sentence.
   * @param sentence
   * @param extendedFeatureSet
   * @return a list of labeled contexts
   */
  def extract(sentence: Sentence, extendedFeatureSet: Set[String] = Set("pos", "chunk", "regexp")): List[LabeledContext] = {
    for (position <- (0 until sentence.length).toList) yield extract(sentence, position, extendedFeatureSet)
  }
  
  def main(args: Array[String]): Unit = {
    val t1 = Token("Barack", Map(Label.PartOfSpeech -> "Np", Label.Chunk -> "I-NP", Label.NamedEntity -> "I-PER"))
    val t2 = Token("Obama", Map(Label.PartOfSpeech -> "Np", Label.NamedEntity -> "I-PER"))
    val t3 = Token("likes", Map(Label.PartOfSpeech -> "V", Label.NamedEntity -> "O"))
    val t4 = Token("Abu", Map(Label.PartOfSpeech -> "N", Label.NamedEntity -> "O"))
    val t5 = Token("Dhabi", Map(Label.PartOfSpeech -> "N", Label.NamedEntity -> "O"))
    
    // val extendedFeatureSet = Set[String]("pos", "chunk", "regexp")
    val extendedFeatureSet = Set.empty[String]

    var sentence = Sentence(ListBuffer(t1, t2, t3, t4, t5))
    var context = extract(sentence, 0, extendedFeatureSet)
    println(context)
    
    val words = ListBuffer("UNBD", "tỉnh", "Cà_Mau", "đã", "xây", 
        "TP", "tiểu_học", "Hữu_Nghị", "báo", "Tuổi_Trẻ", "đưa", 
        "HĐND", "thành_phố", "Hà_Nội", "gặp", "UBND", "TP.", "HCM")
    sentence = Sentence(words.map(w => Token(w, Map[Label.Value, String]())))
    context = extract(sentence, 0, extendedFeatureSet)
    println(context)
  }
  
}