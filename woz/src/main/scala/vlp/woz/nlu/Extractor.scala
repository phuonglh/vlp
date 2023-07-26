package vlp.woz.nlu

import org.apache.spark.SparkConf
import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object Extractor {
  def readFPT(spark: SparkSession, path: String, save: Boolean = false): DataFrame = {
    val df = spark.read.format("com.databricks.spark.xml").option("rootTag", "TEI.DIALOG").option("rowTag", "utterance").load(path)
    df.printSchema()
    // extract the samples: (utteranceId, utterance, Seq[communicativeFunction])
    val rdd = df.rdd.map { row =>
      Row(
        row.getAs[Row]("txt").getAs[Long]("_id").toString,
        row.getAs[Row]("txt").getAs[String]("_VALUE"),
        row.getAs[Seq[Row]]("act").map(_.getAs[String]("domain")),
        row.getAs[Seq[Row]]("act").flatMap(_.getAs[Seq[String]]("communicativeFunction")),
        row.getAs[Seq[Row]]("act").flatMap { row =>
          val j = row.fieldIndex("slot")
          val ss = if (!row.isNullAt(j)) row.getAs[String](j).trim else ""
          if (ss.nonEmpty) ss.replaceAll("""[^\S\r\n]+""", " ") // all spaces but not newline character
            .replaceAll("\"", "")
            .split(",$") // the comma at the end of a line is for separating slot/value pairs
          else
            List.empty[String]
        }
      )
    }
    val schema = StructType(Seq(
      StructField("turnId", StringType, true),
      StructField("utterance", StringType, true),
      StructField("domains", ArrayType(StringType, true), true),
      StructField("acts", ArrayType(StringType, true), true),
      StructField("slots", ArrayType(StringType, true), true)
    ))
    val ef = spark.createDataFrame(rdd, schema)
    if (save) ef.repartition(1).write.json(s"dat/vie/nlu/")
    ef
  }

  /**
   * Groups slot types into a map.
   * @param a slot element
   * @return a map of slotType -> Seq(slotValues)
   */
  private val groupSlots: Array[String] => Map[String, Seq[String]] = (a: Array[String]) => {
    a.map { c =>
      val d = c.split(":")
      if (d.nonEmpty) (d(0).trim, d(1).trim) else ("", "")
    }.groupBy(_._1).mapValues(_.map(_._2))
  }
  val f = udf(groupSlots)

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[4]")
    val spark = SparkSession.builder.config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val df = readFPT(spark, "dat/fpt/0001-0100.xml", false)
    df.show()
    df.printSchema()
    println(df.count())
    df.select("slots").show(false)
    val ef = df.withColumn("state", f(col("slots")))
    ef.select("turnId", "utterance", "state").show()
    ef.printSchema()
    // get all different slot types of the dataset
    import spark.implicits._
    val states = ef.select("state").flatMap { row =>
      val kv = row.getAs[Map[String, Array[String]]](0)
      kv.keys.toList
    }.toDF("slotTypes").distinct()
    states.show(false)
    println(s"Number of different slot types = ${states.count}.")
//    val schema = StructType(Seq(
//      StructField("productName", StringType, true),
//      StructField("productProvider", StringType, true),
//      StructField("productCondition", StringType, true),
//      StructField("productPrice", StringType, true),
//      StructField("productSource", StringType, true),
//      StructField("productColor", StringType, true),
//      StructField("sysNumber", StringType, true)
//    ))
//    df.select("turnId", "slots").show(false)
//    val ef = df.withColumn("fs", from_json(element_at(col("slots"), 1), schema))
//    ef.show()
//    ef.printSchema()
    spark.close()
  }

}
