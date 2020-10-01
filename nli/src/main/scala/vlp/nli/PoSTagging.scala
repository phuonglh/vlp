package vlp.nli

import scala.util.parsing.json._
import org.apache.spark.sql.SparkSession
import org.json4s.jackson.Serialization
import org.json4s._
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.nio.charset.StandardCharsets
import org.apache.spark.ml.PipelineModel
import vlp.tag.ConfigPoS
import vlp.tag.Tagger

/**
  * Part-of-speech tagging of the Vietnamese NLI corpus.
  * phuonglh, September 2020.
  */
object PoSTagging {
  def main(args: Array[String]): Unit = {

    val inputPath = "dat/nli/XNLI-1.0/vi.tok.jsonl"
    val outputPath = "dat/nli/XNLI-1.0/vi.tag.jsonl"

    val s = scala.io.Source.fromFile(inputPath).getLines().toList
    val elements = s.map(x => JSON.parseFull(x).get.asInstanceOf[Map[String,Any]])

    val tagElements = elements.par.map { element => 
      val premise = element("sentence1_tokenized").toString()
      val hypothesis = element("sentence2_tokenized").toString()
      val label = element("gold_label").toString()
      (premise, hypothesis, label)
    }.toList
    val sparkSession = SparkSession.builder().master("local[*]").getOrCreate()

    val (premises, hypotheses) = (tagElements.map(_._1), tagElements.map(_._2))
    val configPoS = ConfigPoS()
    val tagger = new Tagger(sparkSession, configPoS)
    val model = PipelineModel.load(configPoS.modelPath)
    val premisesOutput = tagger.tag(model, premises).map(_.map(p => p._1 + "/" + p._2).mkString(" "))
    val hypothesesOutput = tagger.tag(model, hypotheses).map(_.map(p => p._1 + "/" + p._2).mkString(" "))
    sparkSession.stop()

    val outputElements = (0 until tagElements.size) map { i => 
      Map[String, String](
        "gold_label" -> tagElements(i)._3,
        "sentence1_tokenized" -> tagElements(i)._1,
        "sentence2_tokenized" -> tagElements(i)._2,
        "sentence1_tagged" -> premisesOutput(i),
        "sentence2_tagged" -> hypothesesOutput(i)
      )
    }
    // write JSONL file
    import scala.collection.JavaConversions._
    implicit val formats = Serialization.formats(NoTypeHints)
    val content = outputElements.map(e => Serialization.write(e))
    Files.write(Paths.get(outputPath), content, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    // write JSON file with pretty format
    val prettyContent = outputElements.map(e => Serialization.writePretty(e) + ",")
    val prettyOutput = List("[") ++ prettyContent ++ List("]")
    Files.write(Paths.get(outputPath.substring(0, outputPath.size-1)), prettyOutput, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }
}
