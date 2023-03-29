package vlp.jsl

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame

object T5 {
  val spark: SparkSession = SparkSession.builder.appName("vlp.jsl.T5").master("local[*]").getOrCreate
  import spark.implicits._

  def closeBookQA(pipeline: Pipeline): DataFrame = {
    val data = Seq(
      "Who is the president of Vietnam?", 
      "Who is the president of USA?", 
      "Who is the president of China?", 
      "What is the most common language in India?",
      "What is the capital of France?"
    ).toDF("text").repartition(1)
    pipeline.fit(data).transform(data)
  }

  def openBookQA1(pipeline: Pipeline): DataFrame = {
    val context = "context: Peters last week was terrible! He had an accident and broke his leg while skiing!"
    val data = Seq(
      "question: Why was peters week so bad? " + context,
      "question: How did peters broke his leg? " + context, 
      "question: What did Peters do? " + context,
      "question: How did Peters feel? " + context
    ).toDF("text").repartition(1)
    pipeline.fit(data).transform(data)
  }

  def openBookQA2(pipeline: Pipeline): DataFrame = {
    val context = """context: Alibaba Group founder Jack Ma has made his first appearance since Chinese regulators cracked down on his business empire.
        His absence had fuelled speculation over his whereabouts amid increasing official scrutiny of his businesses.
        The billionaire met 100 rural teachers in China via a video meeting on Wednesday, according to local government media.
        Alibaba shares surged 5% on Hong Kong's stock exchange on the news.      
      """
    val data = Seq(
      "question: Who is Jack ma? " + context,
      "question: Who is founder of Alibaba Group? " + context,
      "question: When did Jack Ma re-appear? " + context,
      "question: How did Alibaba stocks react? " + context,
      "question: Whom did Jack Ma meet? " + context,
      "question: Who did Jack Ma hide from? " + context
    ).toDF("text").repartition(1)
    pipeline.fit(data).transform(data)
  }

  def main(args: Array[String]): Unit = {
    // spark.sparkContext.setLogLevel("ERROR")
    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val t5 = T5Transformer.pretrained("t5_base", "en") 
      .setInputCols("document")
      .setMinOutputLength(10)
      .setMaxOutputLength(400)
      .setDoSample(false)
      .setTopK(50)
      .setTemperature(1.0)
      .setNoRepeatNgramSize(3)
      .setOutputCol("generation")
      .setTask("question")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val df = closeBookQA(pipeline)
    df.select("text", "generation.result").show(false)

    val ef = openBookQA1(pipeline)
    ef.select("generation.result").show(false)

    val ff = openBookQA2(pipeline)
    ff.select("generation.result").show(false)

    spark.stop()
  }

}