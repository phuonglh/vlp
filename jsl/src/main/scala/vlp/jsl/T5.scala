package vlp.jsl

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

object T5 {
  val spark: SparkSession = SparkSession.builder.appName("vlp.jsl.T5").master("local[*]").getOrCreate
  import spark.implicits._

  def main(args: Array[String]): Unit = {
    spark.sparkContext.setLogLevel("ERROR")
    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("documents")
    val t5 = T5Transformer.pretrained("t5_envit5_translation", "xx") 
      .setInputCols(Array("documents"))
      .setMinOutputLength(10)
      .setMaxOutputLength(50)
      .setDoSample(false)
      .setTopK(50)
      .setTemperature(1.0)
      .setNoRepeatNgramSize(3)
      .setOutputCol("generation")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val data = Seq("My name is Leonardo.", "Tôi là Trương Gia Bình, chủ tịch công ty cổ phần FPT.").toDF("text").repartition(1)
    val result = pipeline.fit(data).transform(data)
    result.printSchema
    result.show()
    result.select("generation.result").show(false)
    
    spark.stop()
  }

}