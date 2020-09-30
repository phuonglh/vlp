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
  * phuonglh
  */
object PoSTagging {
  def main(args: Array[String]): Unit = {

    val jsonlPath = "dat/nli/XNLI-1.0/vi.tok.jsonl"
    val s = scala.io.Source.fromFile(jsonlPath).getLines().toList
    val elements = s.map(x => JSON.parseFull(x).get.asInstanceOf[Map[String,Any]])

    val tagElements = elements.par.map { element => 
      val premise = element("sentence1_tokenized").toString()
      val hypothesis = element("sentence2_tokenized").toString()
      (premise, hypothesis)
    }.toList
    val sparkSession = SparkSession.builder().master("local[*]").getOrCreate()

    val (premises, hypotheses) = (tagElements.map(_._1), tagElements.map(_._2))
    val configPoS = ConfigPoS()
    val tagger = new Tagger(sparkSession, configPoS)
    val model = PipelineModel.load(configPoS.modelPath)
    val premisesOutput = tagger.tag(model, premises).map(_.mkString(" "))
    val hypothesesOutput = tagger.tag(model, hypotheses).map(_.mkString(" "))

    import scala.collection.JavaConversions._
    Files.write(Paths.get(jsonlPath + ".p"), premisesOutput, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    Files.write(Paths.get(jsonlPath + ".h"), hypothesesOutput, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    sparkSession.stop()
  }
}
