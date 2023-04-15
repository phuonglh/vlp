package vlp.asa

import java.nio.file.{Files, Paths, StandardOpenOption}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._



case class Entity(text: String, start: Long, end: Long, labels: Array[String], reference: String)
case class AspectSentiment(aspect: String, sentiment: String, reference: String)
case class Element(
  id: String,
  `type`: String,
  text: String,
  tags: Array[String],
  entities: Array[Entity],
  aspect_sentiments: Array[AspectSentiment],
)
case class Sample(
  id: String,
  `type`: String,
  text: String,
  tags: Array[String],
  entities: Array[Entity],
  aspect_sentiments: Array[AspectSentiment],
  comments: Array[Element]
)

case class Instance(postText: String, postSentiments: Array[String], commentText: String, commentSentiments: Array[String])

object SSI {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.getOrCreate()
    import spark.implicits._
    val path = "dat/ssi/project-28-at-2023-02-21-10-28-518a1b2d.json-formatted-4.json"
    val df = spark.read.option("multiline", "true").json(path).as[Sample]
    // flat map the dataset based on comments
    def f(ass: Array[AspectSentiment]) = ass.map(e => e.aspect + "-" + e.sentiment)
    val af = df.flatMap(sample => sample.comments.map(c => (sample.text, f(sample.aspect_sentiments), c.text, f(c.aspect_sentiments))))
      .toDF("postText", "postSentiments", "commentText", "commentSentiments")
    // remove samples having empty sentiment annotation (both in comment and post)
    val bf = af.filter(size(col("postSentiments")) > 0 || size(col("commentSentiments")) > 0)
    spark.stop()
  }
}
