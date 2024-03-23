package vlp.dep

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.sql.SparkSession

import java.io.{BufferedInputStream, FileInputStream}
import java.nio.file.{Files, Paths, StandardOpenOption}
import java.util.zip.GZIPInputStream
import scala.io.Source
import scala.collection.JavaConverters._

// phuonglh@gmail.com
object VocabMaker {

  /**
   * Reads a large embedding file (GloVe, Numberbatch, Fasttex, etc.) and filters
   * words using a dependency vocab, writes result to a new embedding file.
   * @param depVocab
   * @param embeddingPath
   * @param outputPath
   * @return
   */
  def vocabFilter(depVocab: Set[String], embeddingPath: String, outputPath: String) = {
    val lines = if (!embeddingPath.endsWith(".gz")) {
      Source.fromFile(embeddingPath, "UTF-8").getLines()
    } else {
      val is = new GZIPInputStream(new BufferedInputStream(new FileInputStream(embeddingPath)))
      Source.fromInputStream(is).getLines()
    }
    val filtered = if (!embeddingPath.contains("numberbatch-vi")) {
      // English Numberbatch
      lines.map { line =>
        val j = line.indexOf(" ")
        val word = line.substring(0, j)
        val rest = line.substring(j).trim
        (word, rest)
      }.filter(p => depVocab.contains(p._1)).map(p => p._1 + " " + p._2).toList
    } else {
      // Vietnamese Numberbatch, need to remove prefix "/c/vi/"
      lines.map { line =>
        val j = line.indexOf(" ")
        val word = line.substring(0, j).substring(6) // skip the common prefix
        val rest = line.substring(j).trim
        (word, rest)
      }.filter(p => depVocab.contains(p._1)).map(p => p._1 + " " + p._2).toList
    }
    Files.write(Paths.get(outputPath), filtered.asJava, StandardOpenOption.CREATE_NEW, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def main(args: Array[String]): Unit = {
    val language = args(0) // "eng", "ind", "vie"
    if (Set("eng", "ind", "vie").contains(language)) {
      val spark = SparkSession.builder().master("local[*]").getOrCreate()
      val pipelinePath = s"bin/dep/${language}-pre"
      val pipeline = PipelineModel.load(pipelinePath)
      val depVocab = pipeline.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.toSet
      language match {
        case "eng" =>
          val embeddingPath = "dat/emb/numberbatch-en-19.08.txt"
          vocabFilter(depVocab, embeddingPath, "dat/emb/numberbatch-en-19.08.vocab.txt")
//          val embeddingPath = "dat/emb/glove.6B.100d.txt"
//          vocabFilter(depVocab, embeddingPath, "glove.6B.100d.vocab.txt")
//          val embeddingPath = "dat/emb/cc.en.300.vec"
//          vocabFilter(depVocab, embeddingPath, "dat/emb/cc.en.300.vocab.vec")
        case "ind" =>
          val embeddingPath = "dat/emb/cc.id.300.vec"
          vocabFilter(depVocab, embeddingPath, "dat/emb/cc.id.300.vocab.vec")
        case "vie" => // Vietnamese fastText/ConceptNet embeddings
//          val embeddingPath = "dat/emb/cc.vi.300.vec"
//          vocabFilter(depVocab, embeddingPath, "dat/emb/cc.vi.300.vocab.vec")
          val embeddingPath = "dat/emb/numberbatch-vi-19.08.txt"
          vocabFilter(depVocab, embeddingPath, "dat/emb/numberbatch-vi-19.08.vocab.txt")
      }
      spark.stop()
    } else {
      println("Unsupported language: " + args(0))
    }
  }
}
