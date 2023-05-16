package vlp.vsc

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{size, col, avg}
import org.apache.spark.ml.feature._

import org.json4s._
import org.json4s.jackson.Serialization

import org.apache.log4j.{Logger, Level}
import org.slf4j.LoggerFactory
import java.nio.file.{Files, Paths, StandardOpenOption}


case class Stats(
  language: String,
  split: String,
  size: Long,
  averageLength: Double,
  lengthCount: Map[Int, Long],
  vocabSize: Int,
  vocabulary: Array[String]
)

/**
 * Corpus statistics.
 * 
 * phuonglh@gmail.com, Feb. 4, 2023.
 * 
*/
object DataStats {
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  val logger = LoggerFactory.getLogger(getClass.getName)
  implicit val formats = Serialization.formats(NoTypeHints)
  val sc = new SparkConf().setMaster("local[2]")
  val spark = SparkSession.builder.config(sc).getOrCreate()
  import spark.implicits._

  def compute(df: DataFrame, language: String, split: String) = {
    val tokenizer = new Tokenizer().setInputCol("x").setOutputCol("ts")
    val af = tokenizer.transform(df)
    // compute the size of the ts array
    val bf = af.withColumn("n", size(col("ts")))
    // compute the average length of input sentences
    val averageLength = bf.agg(avg("n")).first.get(0).asInstanceOf[Double]
    // length histogram: (n -> count) map
    val cf = bf.groupBy("n").count().sort("n")
    val lengthCount = cf.map(row => (row.getInt(0), row.getLong(1))).collect().toMap
    // unique lowercased tokens in the corpus
    val vectorizer = new CountVectorizer().setInputCol("ts").setOutputCol("v").setMinDF(1)
    val vectorizerModel = vectorizer.fit(af)
    val vocabulary = vectorizerModel.vocabulary
    Stats(language, split, af.count, averageLength, lengthCount, vocabulary.size, vocabulary)
  }

  def write(sc: SparkContext, language: String)(path: String): Unit = {
    val (trainPath, validPath) = VSC.dataPaths(language)
    val (trainDF, validDF) = (DataReader.readDataGED(sc, trainPath), DataReader.readDataGED(sc, validPath))
    val trainStats = compute(trainDF, language, "train")
    var content = Serialization.writePretty(trainStats) + ",\n"
    Files.write(Paths.get(path), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    val validStats = compute(validDF, language, "valid")
    content = Serialization.writePretty(validStats) + ",\n"
    Files.write(Paths.get(path), content.getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
  }

  def main(args: Array[String]): Unit = {
    val languages = Seq("czech", "english", "german", "italian", "swedish")
    for (language <- languages)
      write(spark.sparkContext, language)("dat/vsc/ged-stats.json")
  }
}
