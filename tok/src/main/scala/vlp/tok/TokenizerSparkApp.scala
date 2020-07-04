package vlp.tok

import org.apache.spark.sql.SparkSession
import scopt.OptionParser
import java.io.File
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{Row, DataFrame}
import org.apache.spark.sql.SaveMode

case class ConfigTokenizer(
  master: String = "local[*]",
  executorMemory: String = "8g",
  minPartitions: Int = 1,
  input: String = "",
  format: String = "json", // json/text/vne
  output: String = ""
)

/**
  * phuonglh@gmail.com
  * 
  */
object TokenizerSparkApp {

  def readText(sparkSession: SparkSession, config: ConfigTokenizer): DataFrame = {
    val rdd = if (new File(config.input).isFile()) 
      sparkSession.sparkContext.textFile(config.input, config.minPartitions)
    else sparkSession.sparkContext.wholeTextFiles(config.input, config.minPartitions).map(_._2)

    val rows = rdd.flatMap(content => content.split("""\n+""")).map(Row(_))
    val schema = new StructType().add(StructField("content", StringType, true))
    sparkSession.createDataFrame(rows, schema)
  }

  def readJson(sparkSession: SparkSession, config: ConfigTokenizer): DataFrame = {
    sparkSession.read.json(config.input)
  }

  def readVNE(sparkSession: SparkSession, config: ConfigTokenizer): DataFrame = {
    val rdd = sparkSession.sparkContext.textFile(config.input, config.minPartitions)
    val rows = rdd.map(line => {
      val j = line.indexOf('\t')
      val category = line.substring(0, j).trim
      val content = line.substring(j+1).trim
      Row(category, content)
    })
    val schema = new StructType().add(StructField("category", StringType, true)).add(StructField("content", StringType, true))
    sparkSession.createDataFrame(rows, schema)
  }

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[ConfigTokenizer]("vlp.tok") {
      head("vlp.tok.TokenizerSparkApp", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('e', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[Int]('p', "partitions").action((x, conf) => conf.copy(minPartitions = x)).text("min partitions")
      opt[String]('i', "input").action((x, conf) => conf.copy(input = x)).text("input path (a text file or a directory of .txt/.json files)")
      opt[String]('f', "inputFormat").action((x, conf) => conf.copy(format = x)).text("input format, default is 'json'")
      opt[String]('o', "output").action((x, conf) => conf.copy(output = x)).text("output path which is a directory containing output JSON files")
    }
    parser.parse(args, ConfigTokenizer()) match {
      case Some(config) =>
        val sparkSession = SparkSession.builder().master(config.master)
          .config("spark.executor.memory", config.executorMemory)
          .getOrCreate()

        val df = config.format match {
          case "json" => readJson(sparkSession, config)
          case "text" => readText(sparkSession, config)
          case "vne" => readVNE(sparkSession, config)
        }
        df.printSchema()
        
        val tokenizer = new TokenizerTransformer().setSplitSentences(true).setInputCol("content").setOutputCol("text")
        val tokenized = tokenizer.transform(df)
        val result = if (config.format == "vne") tokenized.select("category", "text") else tokenized.select("text")
        println(s"Number of texts = ${result.count()}")
        result.write.mode(SaveMode.Overwrite).json(config.output)
        sparkSession.stop()
      case None => 
    }
  }
}
