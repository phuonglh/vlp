package vlp.tdp

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Created by phuonglh on 6/22/17.
  */
object App {
  
  def createSentence: Sentence = {
    val m0 = mutable.Map[Label.Value, String](Label.Id -> "0", Label.PartOfSpeech -> "ROOT", Label.Head -> "-1")
    val t0 = Token("ROOT", m0)
    val m1 = mutable.Map[Label.Value, String](Label.Id -> "1", Label.PartOfSpeech -> "PRON", Label.Head -> "2")
    val t1 = Token("I", m1)
    val m2 = mutable.Map[Label.Value, String](Label.Id -> "2", Label.PartOfSpeech -> "VERB", Label.Head -> "0")
    val t2 = Token("love", m2)
    val m3 = mutable.Map[Label.Value, String](Label.Id -> "3", Label.PartOfSpeech -> "PRON", Label.Head -> "2")
    val t3 = Token("you", m3)
    Sentence(ListBuffer(t0, t1, t2, t3))
  }

  def featurize: Unit = {
    val sentence = createSentence
    println(sentence)

    val stack = new mutable.Stack[String]()
    val queue = new mutable.Queue[String]()
    sentence.tokens.foreach(token => queue.enqueue(token.id))
    val arcs = new ListBuffer[Dependency]()
    val config = Config(sentence, stack, queue, arcs).next("SH")
    println(config)

    val featureMap = Map[FeatureType.Value, Boolean](
      FeatureType.Word -> true, 
      FeatureType.PartOfSpeech -> true)
    val extractor = new FeatureExtractor(true)
    val features = extractor.extract(config)
    println(features)
  }
  
  private def readGraphs: Unit = {
    val graphs = GraphReader.read("/eng/en-ud-dev.conllu")
    println(graphs.size)
    println(graphs(0))
    println()
    println(graphs(1))
    graphs(1).sentence.tokens.foreach(println)
  }
  
  def oracle: Unit = {
    val graphs = GraphReader.read("/eng/tests.conllu")
    val oracle = new Oracle(new FeatureExtractor(false))
    // decode the last graph
    val contexts = oracle.decode(graphs.last)
    contexts.foreach(println)
  }
  
  def main(args: Array[String]): Unit = {
    oracle
  }
}
