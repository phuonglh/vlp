package vlp.ner

import vlp.VLP

import scala.collection.mutable.ListBuffer
import scala.io.Source

object CorpusReader {

  /**
    * Reads a NER corpus in CoNLL-2003 format.
    * @param dataPath
    * @return a list of sentences.
    */
  def readCoNLL(dataPath: String): List[Sentence] = {
    val lines = (Source.fromFile(dataPath, "UTF-8").getLines() ++ List("")).toArray
    val sentences = new ListBuffer[Sentence]()
    val indices = lines.zipWithIndex.filter(p => p._1.trim.isEmpty).map(p => p._2)
    var u = 0
    var v = 0
    for (i <- (0 until indices.length)) {
      v = indices(i)
      if (v > u) { // don't treat two consecutive empty lines
        val s = lines.slice(u, v)
        val tokens = s.map(line => {
          val parts = line.trim.split("\\s+")
          Token(parts(0), Map(Label.PartOfSpeech -> parts(1), Label.Chunk -> parts(2), Label.NamedEntity -> parts(3)))
        })
        sentences.append(Sentence(tokens.toList.to[ListBuffer]))
      }
      u = v + 1
    }
    sentences.toList
  }

  /**
   * Reads a VLSP test file and builds sentences to tag.
   * @param dataPath
   * @return a list of [[Sentence]]
   */
  def readVLSPTest(dataPath: String): List[Sentence] = {
    // read lines of the file and remove lines which contains "<s>"
    val lines = Source.fromFile(dataPath, "UTF-8").getLines().toList.filter {
      line => line.trim != "<s>"
    }
    val sentences = new ListBuffer[Sentence]()
    var tokens = new ListBuffer[Token]()
    for (i <- (0 until lines.length)) {
      val line = lines(i).trim
      if (line == "</s>") {
        if (!tokens.isEmpty) sentences.append(Sentence(tokens))
        tokens = new ListBuffer[Token]()
      } else {
        val parts = line.split("\\s+")
        if (parts.length < 3) 
          VLP.log("Invalid line = " + line)
        else 
          tokens.append(Token(parts(0), Map(Label.PartOfSpeech -> parts(1), Label.Chunk -> parts(2))))
      }
    }
    VLP.log(dataPath + ", number of sentences = " + sentences.length)
    sentences.toList
  }
  
  def main(args: Array[String]): Unit = {
    val path = "dat/ner/vie/vie.test"
    val sentences = readCoNLL(path)
    VLP.log("Number of sentences = " + sentences.length)
    sentences.take(10).foreach(s => VLP.log(s.toString))
    sentences.takeRight(10).foreach(s => VLP.log(s.toString))
  }
}