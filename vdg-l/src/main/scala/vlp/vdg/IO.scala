package vlp.vdg

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import org.apache.log4j.{Level, Logger}

import com.intel.analytics.bigdl.dllib.visualization.ValidationSummary

object IO {
  /**
    * Reads text files from a directory or from a text file, ignore all URL lines.
    * @param sparkContext
    * @param path
    * @return a DataFrame with a single field with header 'text'.
    */
  def readTextFiles(sparkContext: SparkContext, path: String): DataFrame = {
    val rdd = sparkContext.textFile(path).map(_.trim).filter(_.nonEmpty)
    val rowRDD = rdd.filter(!_.startsWith("http")).map(line => Row(line))
    val schema = StructType(Array(StructField("text", StringType, false)))
    val sparkSession = SparkSession.builder().getOrCreate()
    sparkSession.createDataFrame(rowRDD, schema)
  }

  /**
    * Reads JSON files from a directory or from a JSON file.
    * @param sparkContext
    * @param path
    * @return a DataFrame with a single field with header 'text'.
    */
  def readJsonFiles(sparkContext: SparkContext, path: String): DataFrame = {
    case class News(url: String, sentences: List[String])

    val lines = sparkContext.textFile(path).map(_.trim).filter(_.nonEmpty).collect()
    implicit val formats = Serialization.formats(NoTypeHints)
    val jsons = lines.map(json => Serialization.read[News](json))
      .filter(_.sentences.size >= 5)
      .flatMap(_.sentences)
      .filter(s => s.size < 400 && s.size >= 20 && !s.contains("<div") && !s.contains("<table") && !s.contains("</p>"))
    val rowRDD = sparkContext.parallelize(jsons).map(line => Row(line))
    val schema = StructType(Array(StructField("text", StringType, false)))
    val sparkSession = SparkSession.builder().getOrCreate()
    sparkSession.createDataFrame(rowRDD, schema)
  }

  /**
    * Retrieves summary information as readable format.
    * @param tag
    * @param epochs
    * @return an array of (validation) accuracy scores at each epoch.
    */
  def extractAccuracySummary(tag: String = "validation", epochs: Int): Array[Float] = {
    val summary = ValidationSummary(appName = "VDG", logDir = "/tmp/")
    val accuracy = summary.readScalar("TimeDistributedTop1Accuracy").map(_._2).takeRight(epochs)
    val outputPath = "/tmp/vdg/" + tag + "accuracy.txt"
    import scala.collection.JavaConverters._
    Files.write(Paths.get(outputPath), accuracy.map(_.toString).toList.asJava, StandardCharsets.UTF_8)
    accuracy
  }
}
