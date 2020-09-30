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

object Utils {

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
          Files.copy(file, target)
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
    * Converts the XML format of BPO annotated files into 2-col with B-I-O format.
    *
    * @param inputPath
    * @param outputPath
    */
  def convertBPO(inputPath: String, outputPath: String): Unit = {
    val entityMap = Map[String, String]("cq" -> "ORG", "dd" -> "LOC", "ng" -> "PER", "vb" -> "DOC")

    def xml2Column(element: scala.xml.Elem): List[String] = {
      val tokens = element.child.flatMap(node => {
        if (node.getClass().getName().contains("Elem")) {
          val entityType = entityMap.getOrElse(node.label, "UNK")
          val tokens = if (entityType != "DOC") vlp.tok.Tokenizer.tokenize(node.text.trim).map(_._3) else node.text.trim.split("\\s+").toList
          Array(tokens.head + " " + "B-" + entityType) ++ tokens.tail.map(word => word + " " + "I-" + entityType)
        } else {
          val tokens = vlp.tok.Tokenizer.tokenize(node.text.trim).map(_._3)
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

  def main(args: Array[String]): Unit = {
    // rename("/Users/phuonglh/Downloads/annotation-2/", "/Users/phuonglh/vlp/dat/ner/stm")
    // convertHUS("/Users/phuonglh/vlp/dat/ner/stm", "/Users/phuonglh/vlp/dat/ner/lad/")
    convertBPO("/Users/phuonglh/vlp/dat/ner/bpo/man", "/Users/phuonglh/vlp/dat/ner/bpo/man.tsv")
    println("Done.")
  }
}
