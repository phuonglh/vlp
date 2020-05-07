package vlp.tdp

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Created by phuonglh on 6/22/17.
  * 
  * A config in transition-based dependency parsing.
  */
case class Config(sentence: Sentence, stack: mutable.Stack[String], queue: mutable.Queue[String], arcs: ListBuffer[Dependency]) {
  /**
    * Computes the next config given a transition.
    * @param transition a transition
    * @return next config.
    */
  def next(transition: String): Config = {
    if (transition == "SH") {
      stack.push(queue.dequeue())
    } else if (transition == "RE") {
      stack.pop()
    } else if (transition.startsWith("LA")) {
      val u = stack.pop()
      val v = queue.front
      val label = transition.substring(3)
      arcs += Dependency(v, u, label)
    } else if (transition.startsWith("RA")) {
      val u = stack.top
      val v = queue.dequeue()
      stack.push(v)
      val label = transition.substring(3)
      arcs += Dependency(u, v, label)
    }
    Config(sentence, stack, queue, arcs)
  }

  /**
    * Is this config reducible? The stack 
    * always contains at least the ROOT token, therefore, if the stack size is less than 1
    * then this config is not reducible. If the top element on the stack have not had a head yet, 
    * then this config is irreducible; otherwise, it is reducible.
    * @return true or false
    */
  def isReducible: Boolean = {
    if (stack.size < 1) false ; else {
      arcs.exists(d => d.dependent == stack.top)
    }
  }

  /**
    * A condition for RE transition in static oracle algorithm. 
    * If config c = (s|i, j|q, A) and there exists k such that (1) k < i and (2) either (k -> j) 
    * or (j -> k) is an arc; then this config is reducible.
    * @param graph a gold graph
    * @return true or false.
    */
  def isReducible(graph: Graph): Boolean = {
    if (stack.size < 2) false; else {
      val i = stack.top
      val j = queue.front
      val allIds = graph.sentence.tokens.map(token => token.id)
      val iId = allIds.indexOf(i)
      var ok = false
      for (kId <- 0 until iId) {
        if (!ok) {
          val k = allIds(kId)
          if (graph.hasArc(k, j) || graph.hasArc(j, k))
            ok = true
        }
      }
      ok
    }
  }
  
  /**
    * Is this config final?
    * @return true or false
    */
  def isFinal: Boolean = {
    queue.isEmpty || stack.isEmpty
  }

  /**
    * Gets the raw sentence of this config.
    * @return a raw sentence.
    */
  def words: String = {
    "\"" + sentence.tokens.map(t => t.word + "/" + t.id).mkString(" ") + "\""
  }
}
