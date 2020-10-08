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
import java.nio.file.StandardOpenOption

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
   * Reads all TSV files in a directory. For each file, we skip the first sentence and the last sentence since in a LAD 
   * file, the first sentence usually is the title (Cong hoa Xa hoi Chu nghia Viet Nam) and the 2 last sentences can be 
   * ignored. We also remove short sentences which has less than 5 tokens.
  */ 
  def readDirectorySTM(dataPath: String, twoColumns: Boolean = false): List[Sentence] = {
    val dir = new File(dataPath)
    if (dir.exists() && dir.isDirectory()) {
      val files = dir.listFiles().filter(file => file.getName().endsWith("tsv")).toList
      println("Number of files = " + files.size)
      files.flatMap{ path => 
        val ss = readCoNLL(path.toString, twoColumns)
        ss.slice(1, ss.size - 2) // Remove the first and last sentence from the corpus.
        .filter(s => s.tokens.size >= 5) // Remove short sentences
      }
    } else List.empty[Sentence]
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
  
  /**
    * Reads sentences in ViSTM format and writes them to an external files
    * of two-column format.
    *
    * @param dataPath
    * @param outputPath 
    */
  def convertSTM(dataPath: String, outputPath: String): Unit = {
    val labelMap = Map("Organization" -> "ORG", "Location" -> "LOC", "Person" -> "PER", "DocType" -> "DOC", "DocCode" -> "DOC", "Date" -> "DATE")
    val lines = (Source.fromFile(dataPath, "UTF-8").getLines() ++ List("")).toArray
      .filter(line => !line.startsWith("#"))
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
          Token(parts(2), Map(Label.NamedEntity -> parts(3)))
        })
        val filteredTokens = tokens.filterNot(token => token.word.trim() == "|")
        sentences.append(Sentence(filteredTokens.toList.to[ListBuffer]))
      }
      u = v + 1
    }
    // convert label[labelId] to B-I format
    val pattern = raw"(\w+)\[(\d+)\]".r
    val newSentences = sentences.toList.map(sentence =>  {
      val entities = sentence.tokens.map(_.namedEntity).toArray
      var count = 0
      var start = true
      val newEntities = entities.map(entity => {
        entity match {
          case pattern(label, id) => {
            if (id.toInt != count) {
              count = id.toInt
              start = true
            } else start = false
            (if (start) "B-" else "I-") + labelMap(label)
          }
          case _ => "O"
        }
      })
      val newTokens = sentence.tokens.zip(newEntities).map(p => Token(p._1.word, Map(Label.NamedEntity -> p._2)))
      Sentence(newTokens)
    })
    // remove all 1-token sentences and write the result to a file of two-column format
    val texts = newSentences.filter(sentence => sentence.tokens.size > 1).map(sentence => {
      sentence.tokens.map(token => token.word + "\t" + token.namedEntity).mkString("\n") + "\n"
    })
    import scala.collection.JavaConversions._
    Files.write(Paths.get(outputPath), texts, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def main(args: Array[String]): Unit = {
    // val path = "dat/ner/vie/vie.test"
    // val sentences = readCoNLL(path)
    // VLP.log("Number of sentences = " + sentences.length)
    // sentences.take(10).foreach(s => VLP.log(s.toString))
    // sentences.takeRight(10).foreach(s => VLP.log(s.toString))

    // convertVLSP2018("dat/ner/xml/dev/", "dat/ner/two/dev.txt")
    // convertVLSP2018("dat/ner/xml/test/", "dat/ner/two/test.txt")
    // convertVLSP2018("dat/ner/xml/train/", "dat/ner/two/train.txt")

    // HUS-group documents:
    // Reads all 2-col LAD files and write all collected sentences to a files: 'dat/ner/lad.tsv'
    val ss = readDirectorySTM("dat/ner/lad/b1", true)
    println("Number of sentences = " + ss.size)
    val content = ss.map { sentence => 
      sentence.tokens.map(token => token.word + "\t" + token.namedEntity).mkString("\n") + "\n"
    }
    import scala.collection.JavaConversions._
    Files.write(Paths.get("dat/ner/lad-b1.tsv"), content, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }
}
