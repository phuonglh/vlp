package vlp.ner

import com.typesafe.config.Config
import vlp.tok.Unicode
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.scalactic._
import spark.jobserver.api.{JobEnvironment, SingleProblem, SparkJob, ValidationProblem}

import scala.util.Try
import vlp.tok.SentenceDetection

/**
  * A named-entity tagger service, which annotates a text with named entities.
  *
  * phuonglh@gmail.com
  */
object TaggerJob extends SparkJob {

  override type JobData = String
  override type JobOutput = List[String]
  lazy val tagger = new Tagger(SparkSession.builder().getOrCreate(), ConfigNER(twoColumns = true, modelPath = "/opt/models/ner/lad/"))

  override def runJob(sc: SparkContext, runtime: JobEnvironment, data: JobData): JobOutput = {
    val text = Unicode.convert(data)
    val sentences = SentenceDetection.run(text).toList
    val ss = tagger.inference(sentences)
    ss.map(s => s.tokens.map(token => token.word + "/" +  token.namedEntity).mkString(" "))
  }

  override def validate(sc: SparkContext, runtime: JobEnvironment, config: Config): Or[JobData, Every[ValidationProblem]] = {
    Try(Good(config.getString("text"))).getOrElse(Bad(One(SingleProblem("No text parameter!"))))
  }

}
