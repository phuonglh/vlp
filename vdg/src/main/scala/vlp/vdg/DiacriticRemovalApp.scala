package vlp.vdg

import org.apache.spark.sql.SparkSession

case class Text(category: String, text: String)

object DiacriticRemovalApp {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().appName(getClass().getName()).master("local[*]").getOrCreate()
    import sparkSession.implicits._
    val input = sparkSession.sparkContext.textFile("dat/hsd/train.tsv")
      .map(line => Text(line.substring(0, 1), line.substring(2)))
      .toDF.as[Text]
    val removal = new DiacriticRemover().setInputCol("text").setOutputCol("withoutAccent")
    val output = removal.transform(input)
    output.write.json("dat/hsd/withoutAccent")
    sparkSession.stop()
  }
}
