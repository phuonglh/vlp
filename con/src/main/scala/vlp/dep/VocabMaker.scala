package vlp.dep

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.sql.SparkSession

import java.nio.file.{Files, Paths, StandardOpenOption}
import scala.io.Source

object VocabMaker {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").getOrCreate()
    val language = "eng"
    // read the vocab
    val pipelinePath = s"bin/dep/${language}-pre"
    val pipeline = PipelineModel.load(pipelinePath)
    val vocab = pipeline.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.toSet
    // load the word vectors
//    val embeddingPath = "dat/emb/numberbatch-en-19.08.txt"
//    val embeddingPath = "dat/emb/glove.6B.100d.txt"
    val embeddingPath = "dat/emb/cc.en.300.vec"
    val lines = Source.fromFile(embeddingPath, "UTF-8").getLines().toList
    val filtered = lines.map { line =>
      val j = line.indexOf(" ")
      val word = line.substring(0, j)
      val rest = line.substring(j).trim
      (word, rest)
    }.filter(p => vocab.contains(p._1)).map(p => p._1 + " " + p._2)
    // write back the filtered word embeddings
    import scala.collection.JavaConverters._
//    Files.write(Paths.get("dat/emb/numberbatch-en-19.08.vocab.txt"), filtered.asJava, StandardOpenOption.CREATE_NEW)
//    Files.write(Paths.get("dat/emb/glove.6B.100d.vocab.txt"), filtered.asJava, StandardOpenOption.CREATE_NEW)
    Files.write(Paths.get("dat/emb/cc.en.300.vocab.vec"), filtered.asJava, StandardOpenOption.CREATE_NEW)
    spark.stop()
  }
}
