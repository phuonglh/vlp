package vlp.tdp

import scala.collection.mutable.ListBuffer

/**
  * Created by phuonglh on 6/22/17.
  */
class FeatureExtractor(val useJointFeatures: Boolean = false, useSuperTag: Boolean = false) extends Serializable {
  
  val featureMap = Map[FeatureType.Value, Boolean](
    FeatureType.PartOfSpeech -> true,
    FeatureType.Word -> true,
    FeatureType.Dependency -> true,
    FeatureType.SuperTag -> useSuperTag
  )
  
  /**
    * Extracts all features from a parser configuration. 
    * @param config a parser configuration
    * @return a space-separated feature strings
    */
  def extract(config: Config): String = {
    val buffer = new ListBuffer[String]()
    if (featureMap.getOrElse(FeatureType.PartOfSpeech, false)) {
      buffer += Extractor.partOfSpeechStack0(config)
      buffer += Extractor.partOfSpeechQueue0(config)
      buffer += Extractor.partOfSpeechStack1(config)
      buffer += Extractor.partOfSpeechQueue1(config)
      if (useJointFeatures) {
        buffer += Extractor.partOfSpeechStack0Queue0(config)
        buffer += Extractor.partOfSpeechStack0Stack1(config)
        buffer += Extractor.partOfSpeechQueue0Queue1(config)
        buffer += Extractor.partOfSpeechStack0Stack1Queue0(config)
      }
    }
    if (featureMap.getOrElse(FeatureType.Word, false)) {
      buffer += Extractor.wordStack0(config)
      buffer += Extractor.shapeStack0(config)
      buffer += Extractor.wordQueue0(config)
      buffer += Extractor.shapeQueue0(config)
      if (useJointFeatures) {
        buffer += Extractor.wordStack0Queue0(config)
        buffer += Extractor.wordStack0Stack1(config)
        buffer += Extractor.wordQueue0Queue1(config)
        buffer += Extractor.partOfSpeechStack0WordStack0(config)
        buffer += Extractor.partOfSpeechQueue0WordQueue0(config)
        buffer += Extractor.partOfSpeechStack0WordQueue0(config)
        buffer += Extractor.partOfSpeechQueue0WordStack0(config)
      }
    }
    if (featureMap.getOrElse(FeatureType.Lemma, false)) {
      buffer += Extractor.lemmaStack0(config)
      buffer += Extractor.lemmaQueue0(config)
    }
    if (featureMap.getOrElse(FeatureType.Dependency, false)) {
      buffer += Extractor.dependencyStack0(config)
      buffer += Extractor.leftmostChildStack0(config)
      buffer += Extractor.rightmostChildStack0(config)
    }
    if (featureMap.getOrElse(FeatureType.SuperTag, false)) {
      buffer += Extractor.superTagStack0(config)
      buffer += Extractor.superTagQueue0(config)
    }
    buffer.mkString(" ")
  }
}

object Extractor {
  
  private def shape(word: String): String = {
    val s = WordShape.shape(word)
    if (s.isEmpty || s == "lower" || s == "ROOT") ""; else s
  }
  
  /**
    * Part-of-speech of the top element on the config's stack. Suppose that 
    * the stack is always non-empty -- it contains at least the ROOT element.
    * @param config a parsing config.
    * @return a feature string
    */
  def partOfSpeechStack0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    "ts0:" + s0.partOfSpeech
  }

  /**
    * Part-of-speech of the top element on the config's queue.
    * @param config a parsing config
    * @return a feature string
    */
  def partOfSpeechQueue0(config: Config): String = {
    val q0 = config.sentence.token(config.queue.front)
    "tq0:" + q0.partOfSpeech
  }

  /**
    * Word of the top element on the config' stack. Suppose that 
    * the stack is always non-empty -- it contains at least the ROOT element.
    * @param config a parsing config
    * @return a feature string
    */
  def wordStack0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    "ws0:" + s0.word
  }

  /**
    * Word of the top element on the config's queue.
    * @param config a parsing config
    * @return a feature string
    */
  def wordQueue0(config: Config): String = {
    val q0 = config.sentence.token(config.queue.front)
    "wq0:" + q0.word
  }

  /**
    * Shape of the first word on the queue.
    * @param config a parsing context
    * @return a feature string
    */
  def shapeQueue0(config: Config): String = {
    val q0 = config.sentence.token(config.queue.front).word
    "sq0:" + shape(q0)
  }

  /**
    * Shape of the top word on the stack.
    * @param config a parsing context
    * @return a feature string
    */
  def shapeStack0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top).word
    "ss0:" + shape(s0)
  }
  
  /**
    * Lemma of the top element on the config' stack. Suppose that 
    * the stack is always non-empty -- it contains at least the ROOT element.
    * @param config a parsing config
    * @return a feature string
    */
  def lemmaStack0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    "ls0:" + s0.lemma
  }

  /**
    * Lemma of the top element on the config's queue.
    * @param config a parsing config
    * @return a feature string
    */
  def lemmaQueue0(config: Config): String = {
    val q0 = config.sentence.token(config.queue.front)
    "lq0:" + q0.lemma
  }

  /**
    * Part-of-speech of the second-top element on the config's stack. Note that 
    * the stack is always non-empty: it contains at least the ROOT element.
    * @param config a parsing config.
    * @return a feature string
    */
  def partOfSpeechStack1(config: Config): String = {
    if (config.stack.size < 2) "ts1:None"; else {
      val s1 = config.sentence.token(config.stack(1))
      "ts1:" + s1.partOfSpeech
    }
  }

  /**
    * Part-of-speech of the second element on the config's queue.
    * @param config a parsing config
    * @return a feature string
    */
  def partOfSpeechQueue1(config: Config): String = {
    if (config.queue.size < 2) "tq1:None"; else {
      val q1 = config.sentence.token(config.queue(1))
      "tq1:" + q1.partOfSpeech
    }
  }
  
  // Joint features
  //
  /**
    * Parts-of-speech of the top elements on the stack and the queue.
    * @param config a parsing config
    * @return a feature string
    */
  def partOfSpeechStack0Queue0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    val q0 = config.sentence.token(config.queue.front)
    "ts0+tq0:" + s0.partOfSpeech + '+' + q0.partOfSpeech
  }

  /**
    * Shapes of the top elements on the stack and the queue.
    * @param config a parsing config
    * @return a feature string
    */
  def wordStack0Queue0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    val q0 = config.sentence.token(config.queue.front)
    "ws0+wq0:" + s0.word + '+' + q0.word
  }

  /**
    * Parts-of-speech of two elements on the stack.
    * @param config a parsing config
    * @return a feature string
    */
  def partOfSpeechStack0Stack1(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    if (config.stack.size < 2) "ts0+ts1:" + s0.partOfSpeech + "+None"; else {
      val s1 = config.sentence.token(config.stack(1))
      "ts0+ts1:" + s0.partOfSpeech + '+' + s1.partOfSpeech 
    }
  }

  /**
    * Parts-of-speech of two elements on the queue.
    * @param config a parsing config
    * @return a feature string
    */
  def partOfSpeechQueue0Queue1(config: Config): String = {
    val q0 = config.sentence.token(config.queue.front)
    if (config.queue.size < 2) "tq0+tq1:" + q0.partOfSpeech + "+None"; else {
      val q1 = config.sentence.token(config.queue(1))
      "tq0+tq1:" + q0.partOfSpeech + '+' + q1.partOfSpeech
    }
  }

  /**
    * Words of two elements on the stack.
    * @param config a parsing config
    * @return a feature string
    */
  def wordStack0Stack1(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    if (config.stack.size < 2) "ws0+ws1:" + s0.word + "+None"; else {
      val s1 = config.sentence.token(config.stack(1))
      "ws0+ws1:" + s0.word + '+' + s1.word
    }
  }

  /**
    * Word of two elements on the queue.
    * @param config a parsing config
    * @return a feature string
    */
  def wordQueue0Queue1(config: Config): String = {
    val q0 = config.sentence.token(config.queue.front)
    if (config.queue.size < 2) "wq0+wq1:" + q0.word + "+None"; else {
      val q1 = config.sentence.token(config.queue(1))
      "wq0+wq1:" + q0.word + '+' + q1.word
    }
  }

  /**
    * Dependency label of the top element on the stack.
    * @param config a parsing config
    * @return a feature string
    */
  def dependencyStack0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    "ds0:" + s0.dependencyLabel
  }

  /**
    * Joint part-of-speech and word of the top element on the stack.
    * @param config a parsing config
    * @return a feature string
    */
  def partOfSpeechStack0WordStack0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    "ts0+ws0:" + s0.partOfSpeech + '+' + s0.word
  }

  /**
    * Joint part-of-speech and word of the first element on the buffer.
    * @param config a parsing config
    * @return a feature string
    */
  def partOfSpeechQueue0WordQueue0(config: Config): String = {
    val q0 = config.sentence.token(config.queue.front)
    "tq0+wq0:" + q0.partOfSpeech + '+' + q0.word
  }

  /**
    * Parts-of-speech of two elements on the stack plus the first element on 
    * the queue.
    * @param config a parsing config
    * @return a feature string
    */
  def partOfSpeechStack0Stack1Queue0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    val q0 = config.sentence.token(config.queue.front)
    if (config.stack.size < 2) "ts0+ts1+tq0:" + s0.partOfSpeech + "+None" + '+' + q0.partOfSpeech; else {
      val s1 = config.sentence.token(config.stack(1))
      "ts0+ts1+tq0:" + s0.partOfSpeech + '+' + s1.partOfSpeech + '+' + q0.partOfSpeech
    }
  }

  /**
    * Part-of-speech of the top element on the stack and the word of the first 
    * element on the buffer.
    * @param config parsing config
    * @return a feature string
    */
  def partOfSpeechStack0WordQueue0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    val q0 = config.sentence.token(config.queue.front)
    "ts0+wq0:" + s0.partOfSpeech + '+' + q0.word
  }

  /**
    * Part-of-speech of the first element on the queue and 
    * the word of the first element on the stack.
    * @param config parsing context.
    * @return a feature string
    */
  def partOfSpeechQueue0WordStack0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    val q0 = config.sentence.token(config.queue.front)
    "tq0+ws0:" + q0.partOfSpeech + '+' + s0.word
  }

  /**
    * Extracts the left most child of the top element on the stack.
    * @param config a parsing config
    * @return a feature string
    */
  def leftmostChildStack0(config: Config): String = {
    /**
      * Extracts the leftmost child of a node (already in the parsing stack).
      * @param tokenId a token id
      * @return a feature string
      */
    def leftmostChild(tokenId: String): String = {
      // create a list of token ids
      val tokenIds = config.sentence.tokens.map(token => token.id).toList
      // find the position of the given tokenId in that list
      val position = tokenIds.indexWhere(id => id == tokenId)
      if (position > 0) {
        // take the left sub-list of the position and reverse it 
        val leftList = tokenIds.slice(0, position).reverse
        // get all dependents of the given tokenId 
        val dependents = config.arcs.filter(d => d.head == tokenId).map(d => d.dependent)
        if (!dependents.isEmpty) {
          // find the leftmost dependent
          var result = "None"
          for (j <- 0 until leftList.length)
            if (dependents.contains(leftList(j)) && result == "None") {
              result = config.sentence.token(leftList(j)).word
            }
          result
        } else "None"
      } else "None"
    }
    
    "lcs0:" + leftmostChild(config.stack.top)
  }
  
  def rightmostChildStack0(config: Config): String = {
    /**
      * Extracts the rightmost child of a node (already in the parsing stack).
      * @param tokenId a token id
      * @return a feature string
      */
    def rightmostChild(tokenId: String): String = {
      // create a list of token ids
      val tokenIds = config.sentence.tokens.map(token => token.id).toList
      // find the position of the given tokenId in that list
      val position = tokenIds.indexWhere(id => id == tokenId)
      if (position > 0) {
        // take the right sub-list of the position
        val rightList = tokenIds.slice(position, tokenIds.length)
        // get all dependents of the given tokenId 
        val dependents = config.arcs.filter(d => d.head == tokenId).map(d => d.dependent)
        if (!dependents.isEmpty) {
          // find the leftmost dependent
          var result = "None"
          for (j <- 0 until rightList.length)
            if (dependents.contains(rightList(j)) && result == "None") {
              result = config.sentence.token(rightList(j)).word
            }
          result
        } else "None"
      } else "None"      
    }
    
    "rcs0:" + rightmostChild(config.stack.top)
  }

  /**
    * Super tag of the top word on stack.
    * @param config a parsing config
    * @return a feature string
    */
  def superTagStack0(config: Config): String = {
    val s0 = config.sentence.token(config.stack.top)
    "sts0:" + s0.superTag
  }

  /**
    * Super tag of the first word on queue.
    * @param config a parsing config
    * @return a feature string
    */
  def superTagQueue0(config: Config): String = {
    val q0 = config.sentence.token(config.queue.front)
    "stq0:" + q0.superTag
  }
  
}
