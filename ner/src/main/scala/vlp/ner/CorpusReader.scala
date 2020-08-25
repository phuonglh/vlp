package vlp.ner

import vlp.VLP

import scala.collection.mutable.ListBuffer
import scala.io.Source
import java.nio.file.Files
import java.nio.file.Paths
import java.util.stream.Collectors
import java.io.File
import java.nio.charset.StandardCharsets
import java.{util => ju}

object CorpusReader {

  /**
    * Reads a NER corpus in CoNLL-2003 format.
    * @param dataPath
    * @return a list of sentences.
    */
  def readCoNLL(dataPath: String, twoColumns: Boolean = false): List[Sentence] = {
    val lines = (Source.fromFile(dataPath, "UTF-8").getLines() ++ List("")).toArray
    val sentences = new ListBuffer[Sentence]()
    val indices = lines.zipWithIndex.filter(p => p._1.trim.isEmpty).map(p => p._2)
    var u = 0
    var v = 0
    for (i <- (0 until indices.length)) {
      v = indices(i)
      if (v > u) { // don't treat two consecutive empty lines
        val s = lines.slice(u, v)
        val tokens = if (!twoColumns) s.map(line => {
          val parts = line.trim.split("\\s+")
          Token(parts(0), Map(Label.PartOfSpeech -> parts(1), Label.Chunk -> parts(2), Label.NamedEntity -> parts(3)))
        }) else s.map(line => {
          val parts = line.trim.split("\\s+")
          Token(parts(0), Map(Label.NamedEntity -> parts(1)))
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

  /**
    * Reads sentences in raw text of VLSP 2018 dataset which use XML format and writes them to an external files
    * of two-column format.
    *
    * @param dataPath
    * @param outputPath 
    */
  def convertVLSP2018(dataPath: String, outputPath: String): Unit = {

    def xml2Column(element: scala.xml.Elem): List[String] = {
      val tokens = element.child.flatMap(node => {
        if (node.getClass().getName().contains("Elem")) {
          val entityType = (node \ "@TYPE").theSeq.head.text
          val words = vlp.tok.Tokenizer.tokenize(node.text.trim).map(_._3)
          Array(words.head + " " + "B-" + entityType) ++ words.tail.map(word => word + " " + "I-" + entityType)
        } else {
          val words = vlp.tok.Tokenizer.tokenize(node.text.trim).map(_._3)
          words.map(word => word + " " + "O")
        }
      })
      tokens.toList
    }

    // get all files in the data path
    import scala.collection.JavaConversions._
    val paths = Files.walk(Paths.get(dataPath)).collect(Collectors.toList()).filter(path => Files.isRegularFile(path))
    println(paths.size)
    // read all lines of these files and collect them into a list of lines
    val lines = paths.flatMap(path => {
      try {
        Source.fromFile(new File(path.toString()), "utf-8").getLines().map(_.trim()).filter(_.nonEmpty).toList
      } catch {
        case e: Exception => { println(path.toString); List.empty }
      }
    })
    val elements = lines.toList.map(line => {
      // println(line)
      scala.xml.XML.loadString("<s>" + line.replaceAll("&", "&amp;") + "</s>")
    })
    val texts = elements.map(element => xml2Column(element).mkString("\n") + "\n")
    Files.write(Paths.get(outputPath), texts, StandardCharsets.UTF_8)
  }
  
  def main(args: Array[String]): Unit = {
    val path = "dat/ner/vie/vie.test"
    val sentences = readCoNLL(path)
    VLP.log("Number of sentences = " + sentences.length)
    sentences.take(10).foreach(s => VLP.log(s.toString))
    sentences.takeRight(10).foreach(s => VLP.log(s.toString))

    convertVLSP2018("dat/ner/xml/dev/", "dat/ner/two/dev.txt")
    convertVLSP2018("dat/ner/xml/test/", "dat/ner/two/test.txt")
    convertVLSP2018("dat/ner/xml/train/", "dat/ner/two/train.txt")
  }
}