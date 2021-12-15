package vlp.ner

import java.nio.file.Files
import java.nio.file.FileVisitor
import java.nio.file.FileVisitResult
import java.nio.file.attribute.BasicFileAttributes
import java.nio.file.Path
import java.nio.file.Paths
import java.io.IOException
import java.nio.file.SimpleFileVisitor
import java.util.stream.Collectors
import scala.io.Source
import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption

object Utils {
  final val tokenizer = new vlp.tok.Tokenizer()
  /**
    * Renames all files in the STM annotation directory: "1.txt/lanvy.tsv" should be renamed to "1-lanvy.tsv"
    *
    * @param inputPath
    * @param outputPath
    */
  def rename(inputPath: String, outputPath: String): Unit = {
    val files = Files.walkFileTree(Paths.get(inputPath), new FileVisitor[Path]() {
      var id = ""
      def preVisitDirectory(dir: Path, attrs: BasicFileAttributes): FileVisitResult = {
        val currentDirName = dir.getFileName().toString()
        if (currentDirName.endsWith(".txt")) {
          id = currentDirName.substring(0, currentDirName.indexOf("."))
        } else {
          println("Current dir = " + id + " from " + currentDirName)
        }
        FileVisitResult.CONTINUE
      }
      def visitFile(file: Path, attrs: BasicFileAttributes): FileVisitResult = {
        val currentFileName = file.getFileName().toString()
        if (id.nonEmpty && currentFileName.endsWith("tsv")) {
          val target = Paths.get(outputPath, id + "-" + currentFileName)
          Files.copy(file, target, StandardCopyOption.REPLACE_EXISTING)
        }
        FileVisitResult.CONTINUE
      }
      def postVisitDirectory(dir: Path, exc: IOException): FileVisitResult = FileVisitResult.CONTINUE
      def visitFileFailed(file: Path, exc: IOException): FileVisitResult = {
        println("Failed to access file: " + file.toString())
        FileVisitResult.TERMINATE
      }
    })
  }

  /**
    * Converts the 4-col format of STM annotated files into 2-col with B-I-O format.
    *
    * @param inputPath
    * @param outputPath
    */
  def convertHUS(inputPath: String, outputPath: String): Unit = {
    val files = Files.walkFileTree(Paths.get(inputPath), new SimpleFileVisitor[Path]() {
      override def visitFile(file: Path, attrs: BasicFileAttributes): FileVisitResult = {
        val currentFileName = file.getFileName().toString()
        println(currentFileName)
        if (currentFileName.toString().endsWith("tsv")) {
          val target = Paths.get(outputPath, currentFileName).toString()
          CorpusReader.convertSTM(file.toString(), target)
        }
        FileVisitResult.CONTINUE
      }
    })
  }

  /**
    * Converts the XML format of BPO annotated files into 2-col with B-I-O format; word segmentation is performed 
    * during the conversion.
    *
    * @param inputPath
    * @param outputPath
    */
  def convertBPO(inputPath: String, outputPath: String): Unit = {
    val entityMap = Map[String, String]("cq" -> "ORG", "dd" -> "LOC", "ng" -> "PER", "vb" -> "DOC", "date" -> "DATE")

    def xml2Column(element: scala.xml.Elem): List[String] = {
      val tokens = element.child.flatMap(node => {
        if (node.getClass().getName().contains("Elem")) {
          val entityType = entityMap.getOrElse(node.label, "UNK")
          val tokens = if (entityType != "DOC") tokenizer.tokenize(node.text.trim).map(_._3) else node.text.trim.split("\\s+").toList
          Array(tokens.head + " " + "B-" + entityType) ++ tokens.tail.map(word => word + " " + "I-" + entityType)
        } else {
          val tokens = tokenizer.tokenize(node.text.trim).map(_._3)
          tokens.map(token => token + " " + "O")
        }
      })
      tokens.toList
    }

    // get all files in the data path
    import scala.collection.JavaConversions._
    val paths = Files.walk(Paths.get(inputPath)).collect(Collectors.toList()).filter(path => Files.isRegularFile(path))
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
    * Converts a file of 2-col BIO format into XML format (as used in BPO data).
    *
    * @param inputPath
    * @param outputPath
    */
  def bio2Xml(inputPath: String, outputPath: String): Unit = {
    val typMap = Map("B-ORG" -> "cq", "I-ORG" -> "cq", "B-LOC" -> "dd", "I-LOC" -> "dd", "B-PER" -> "ng", "I-PER" -> "ng", 
      "B-DATE" -> "date", "I-DATE" -> "date", "B-DOC" -> "vb", "I-DOC" -> "vb")

    def convert(sentence: Sentence): String = {
      val pairs = sentence.tokens.toList.map(token => (token.word, typMap.getOrElse(token.namedEntity, "O")))
      val n = pairs.size
      val st = new StringBuffer()
      var u = 0
      var v = 0
      var t = pairs(u)._2
      while (u < n) {
        v = u + 1
        while (v < n && pairs(v)._2 == pairs(u)._2)
          v = v + 1
        val content = pairs.slice(u, v).map(_._1).mkString(" ")
        if (t != "O") {
          st.append("<" + t + ">")
          st.append(content)
          st.append("</" + t + ">")
          st.append(" ")
        } else {
          st.append(content)
          st.append(" ")
        }
        u = v
        if (u < n) t = pairs(u)._2
      }
      st.toString().trim()
    }
    val sentences = CorpusReader.readCoNLL(inputPath, true)
    val xs = sentences.map(s => convert(s)).filter(s => s.indexOf("</") > 0)
    import scala.collection.JavaConversions._
    Files.write(Paths.get(outputPath), xs, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def main(args: Array[String]): Unit = {
    // rename("/Users/phuonglh/Downloads/annotation-3/b2", "/Users/phuonglh/vlp/dat/ner/stm/b2")
    // convertHUS("/Users/phuonglh/vlp/dat/ner/stm/b2", "/Users/phuonglh/vlp/dat/ner/lad/b2")

    convertBPO("/Users/phuonglh/vlp/dat/ner/man/man.txt", "/Users/phuonglh/vlp/dat/ner/man/man.tsv")

    // bio2Xml("dat/ner/lad-b2.tsv", "dat/ner/lad-b2.txt")
    println("Done.")
  }
}
