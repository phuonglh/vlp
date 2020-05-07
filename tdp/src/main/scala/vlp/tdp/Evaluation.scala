package vlp.tdp

import java.util.concurrent.atomic.AtomicInteger

/**
  * Created by phuonglh on 7/1/17.
  */
object Evaluation {
  val uasCounter = new AtomicInteger(0)
  val lasCounter = new AtomicInteger(0)
  val numTokens = new AtomicInteger(0)

  /**
    * Evaluates the accuracy of the parser on a list of graphs.
    * @param parser
    * @param graphs
    */
  def eval(parser: Parser, graphs: List[Graph], countPuncts: Boolean = false): Unit = {
    uasCounter.set(0)
    lasCounter.set(0)
    numTokens.set(0)
    val correct = graphs.map(g => g.sentence)
    val guess = parser.parse(correct)
    for (i <- 0 until correct.length) {
      val c = correct(i)
      val g = guess(i).sentence.tokens
      val x = if (countPuncts) c.tokens else c.tokens.filter(token => token.universalPartOfSpeech != "PUNCT")
      for (j <- 0 until x.length) {
        if (x(j).head == g(j).head) {
          uasCounter.incrementAndGet()
          if (x(j).dependencyLabel == g(j).dependencyLabel)
            lasCounter.incrementAndGet()
        }
      }
      numTokens.addAndGet(x.length)
    }
  }

  /**
    * Unlabeled attachment score.
    * @return a percentage.
    */
  def uas: Double = {
    if (numTokens.get() > 0) uasCounter.get().toDouble / numTokens.get(); else 0
  }
  
  /**
    * Labeled attachment score.
    * @return a percentage.
    */
  def las: Double = {
    if (numTokens.get() > 0) lasCounter.get().toDouble / numTokens.get(); else 0
  }
}
