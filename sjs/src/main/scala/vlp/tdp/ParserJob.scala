package vlp.tdp

import com.typesafe.config.ConfigFactory
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.scalactic._
import scala.util.Try

import spark.jobserver.api.{SparkJob => NewSparkJob, _}
import spark.jobserver.{SparkJob, SparkJobInvalid, SparkJobValid, SparkJobValidation}
import org.apache.spark.sql.SparkSession


/**
 * phuonglh, September 2020.
 * 
 */
object ParserJob extends NewSparkJob {
  type JobData = String
  type JobOutput = Seq[String]

  lazy val parser = new Parser(SparkSession.builder().getOrCreate(), ConfigTDP(modelPath = "/opt/models/tdp/"), ClassifierType.MLR, false)

  def runJob(sc: SparkContext, runtime: JobEnvironment, data: JobData): JobOutput = {
    val graph = parser.parseWithPartOfSpeech(data)
    graph.toString().split("""\n""").toSeq
  }

  def validate(sc: SparkContext, runtime: JobEnvironment, config: com.typesafe.config.Config): JobData Or Every[ValidationProblem] = {
    Try(config.getString("input"))
      .map(sentence => Good(sentence))
      .getOrElse(Bad(One(SingleProblem("No input param"))))
  }
}
