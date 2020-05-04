package vlp.tdp

import java.util.concurrent.atomic.AtomicInteger

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Created by phuonglh on 6/24/17.
  *
  * This is a static oracle for transition-based dependency parsing. 
  * 
  * Decodes a manually-annotated dependency graph for 
  * parsing contexts and their corresponding transitions. This 
  * utility is used to create training data.
  *
  */
class Oracle(val featureExtractor: FeatureExtractor) {
  
  val counter = new AtomicInteger(0)

  /**
    * Derives a transition sequence from this dependency graph. This 
    * is used to reconstruct the parsing process of a sentence.
    * 
    * @param graph a dependency graph
    * @return a list of parsing context
    */
  def decode(graph: Graph): List[Context] = {
    // create the initial config
    val stack = mutable.Stack[String]()
    val queue = mutable.Queue[String]()
    graph.sentence.tokens.foreach(token => queue.enqueue(token.id))
    stack.push(queue.dequeue())
    val arcs = ListBuffer[Dependency]()
    var config = Config(graph.sentence, stack, queue, arcs)
    val contexts = ListBuffer[Context]()
    while (!config.isFinal) {
      // extract features
      val features = featureExtractor.extract(config)
      // extract transition
      val u = stack.top
      val v = queue.front
      var transition = "SH"
      if (graph.hasArc(v, u)) {
        transition = "LA-" + graph.sentence.token(u).dependencyLabel
      } else if (graph.hasArc(u, v)) {
        transition = "RA-" + graph.sentence.token(v).dependencyLabel
      } else {
        if (config.isReducible(graph))
          transition = "RE"
      }
      // add a parsing context
      contexts += Context(counter.getAndIncrement(), features, transition)
      config = config.next(transition)
    }
    contexts.toList
  }

  /**
    * Derives all parsing contexts from a treebank of many dependency graphs.
    * @param graphs a list of manually-annotated dependency graphs.
    * @return a list of parsing contexts.
    */
  def decode(graphs: List[Graph]): List[Context] = {
    graphs.par.flatMap(graph => decode(graph)).toList
  }
}
