package vlp.nli

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.log4j.Logger
import org.apache.log4j.Level

object Statistics {
  def main(arg: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val sparkSession = SparkSession.builder().appName("nli.Statistics").master("local[4]").getOrCreate()
    val inputPath = "dat/nli/SNLI-1.0/snli_1.0_dev.jsonl"
    val df = sparkSession.read.json(inputPath)
    df.printSchema()
    df.show()
    val tokenizer1 = new RegexTokenizer().setInputCol("sentence1").setOutputCol("sentence1_tokenized").setPattern("""[\s,.;?'":]+""")
    val tokenizer2 = new RegexTokenizer().setInputCol("sentence2").setOutputCol("sentence2_tokenized").setPattern("""[\s,.;?'":]+""")
    val alpha = tokenizer2.transform(tokenizer1.transform(df))
    alpha.select("gold_label", "sentence1_tokenized", "sentence2_tokenized").show(false)
    import sparkSession.implicits._
    val count = alpha.select("sentence1_tokenized", "sentence2_tokenized").map(row => (row.getAs[Seq[String]](0).size, row.getAs[Seq[String]](1).size))
    count.show(false)
    val histogram1 = count.groupBy("_1").count().sort($"count".desc)
    val histogram2 = count.groupBy("_2").count().sort($"count".desc)
    histogram1.show(30, false)
    histogram2.show(30, false)

    val labelCount = alpha.select("gold_label").groupBy("gold_label").count()
    labelCount.show(false)
    sparkSession.stop()
  }
}
