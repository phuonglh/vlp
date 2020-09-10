package vlp.ner

import java.nio.file.Files
import java.nio.file.FileVisitor
import java.nio.file.FileVisitResult
import java.nio.file.attribute.BasicFileAttributes
import java.nio.file.Path
import java.nio.file.Paths
import java.io.IOException
import java.nio.file.SimpleFileVisitor

object Utils {

  /**
    * Renames all files in the STM annotation directory: "1.txt/lanvy.tsv" should be renamed to "1-lanvy.txt"
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
  def convert(inputPath: String, outputPath: String): Unit = {
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

  def main(args: Array[String]): Unit = {
    // rename("/Users/phuonglh/Downloads/annotation-2/", "/Users/phuonglh/vlp/dat/ner/stm")
    convert("/Users/phuonglh/vlp/dat/ner/stm", "/Users/phuonglh/vlp/dat/ner/stm/two")
  }
}
