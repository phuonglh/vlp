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
    inputPath: String = "",
    inputFormat: String = "json", // json/text/vne
    inputColumnName: String = "content",
    outputColumnName: String = "text",
    outputPath: String = ""
)

/**
  * phuonglh@gmail.com
  * 
  */
object VietnameseTokenizer {

    def readText(sparkSession: SparkSession, config: ConfigTokenizer): DataFrame = {
        val rdd = if (new File(config.inputPath).isFile()) 
            sparkSession.sparkContext.textFile(config.inputPath, config.minPartitions)
        else sparkSession.sparkContext.wholeTextFiles(config.inputPath, config.minPartitions).map(_._2)

        val rows = rdd.flatMap(content => content.split("""\n+""")).map(Row(_))
        val schema = new StructType().add(StructField("content", StringType, true))
        sparkSession.createDataFrame(rows, schema)
    }

    def readVNE(sparkSession: SparkSession, config: ConfigTokenizer): DataFrame = {
        val rdd = sparkSession.sparkContext.textFile(config.inputPath, config.minPartitions)
        val rows = rdd.map(line => {
            val j = line.indexOf('\t')
            val category = line.substring(0, j).trim
            val content = line.substring(j+1).trim
            Row(category, content)
        })
        val schema = new StructType().add(StructField("category", StringType, true)).add(StructField("content", StringType, true))
        sparkSession.createDataFrame(rows, schema)
    }


    /**
    * Reads a line-oriented JSON file. Number of lines is the number of rows of the resulting data frame.
    *
    * @param sparkSession
    * @param config
    * @return a data frame.
    */
    def readJson(sparkSession: SparkSession, config: ConfigTokenizer): DataFrame = {
        sparkSession.read.json(config.inputPath)
    }

    def readJsonMultiline(sparkSession: SparkSession, path: String): DataFrame = {
        sparkSession.read.option("multiLine", "true").option("mode", "PERMISSIVE").json(path)
    }

    def main(args: Array[String]): Unit = {
        val parser = new OptionParser[ConfigTokenizer]("vlp.tok.Tokenizer") {
        head("vlp.tok.VietnamseTokenizer", "1.0")
        opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
        opt[String]('e', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
        opt[Int]('p', "partitions").action((x, conf) => conf.copy(minPartitions = x)).text("min partitions")
        opt[String]('i', "inputPath").action((x, conf) => conf.copy(inputPath = x)).text("input path (a text file or a directory of .txt/.json files)")
        opt[String]('f', "inputFormat").action((x, conf) => conf.copy(inputFormat = x)).text("input format, default is 'json'")
        opt[String]('u', "inputColumnName").action((x, conf) => conf.copy(inputColumnName = x)).text("input column name, default is 'content'")
        opt[String]('v', "outputColumnName").action((x, conf) => conf.copy(outputColumnName = x)).text("output column name, default is 'text'")
        opt[String]('o', "outputPath").action((x, conf) => conf.copy(outputPath = x)).text("output path which is a directory containing output JSON files")
        }
        parser.parse(args, ConfigTokenizer()) match {
        case Some(config) =>
            val sparkSession = SparkSession.builder().master(config.master).config("spark.executor.memory", config.executorMemory).getOrCreate()

            val df = config.inputFormat match {
                case "json" => readJson(sparkSession, config)
                case "text" => readText(sparkSession, config)
                case "vnexpress" => readVNE(sparkSession, config)
                case "shinra" => readJsonMultiline(sparkSession, config.inputPath)
            }
            df.printSchema()
            
            val tokenizer = new TokenizerTransformer().setSplitSentences(true).setInputCol(config.inputColumnName).setOutputCol(config.outputColumnName)
            val tokenized = tokenizer.transform(df)
            val result = config.inputFormat match {
                case "vnexpress" => tokenized.select("category", "text")
                case "shinra" => tokenized.select("id", config.outputColumnName, "title", "category", "outgoingLink", "redirect", "clazz")
                case _ => tokenized.select("text")
            }
            println(s"Number of texts = ${result.count()}")
            result.repartition(config.minPartitions).write.mode(SaveMode.Overwrite).json(config.outputPath)
            sparkSession.stop()
        case None => 
        }
    }
}
