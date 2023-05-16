package vlp.vsc

import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._

import scala.collection.mutable.ListBuffer
import scala.io.Source
import java.io.FileWriter


object DataReader {
  /**
    * Reads an input text file and creates a data frame of two columns "x, y", where 
    * "x" are input token sequences and "y" are corresponding label sequences. The text file 
    * has a format of line-pair oriented: (y_i, x_i).
    *
    * @param sc a Spark context
    * @param dataPath
    * @return a data frame with two columns (x, y), representing input and output sequence.
    */
  def readData(sc: SparkContext, dataPath: String): DataFrame = {
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()    
    import spark.implicits._
    val df = sc.textFile(dataPath).zipWithIndex.toDF("line", "id")
    val df0 = df.filter(col("id") % 2 === 0).withColumn("y", col("line"))
    val df1 = df.filter(col("id") % 2 === 1).withColumn("x", col("line")).withColumn("id0", col("id") - 1)
    val af = df0.join(df1, df0.col("id") === df1.col("id0"))
    return af.select("x", "y")
  }

  /**
    * Reads a corpus in GED format (two columns, *.tsv)
    * @param sc
    * @param dataPath
    * @return a data frame
    */
  def readDataGED(sc: SparkContext, dataPath: String): DataFrame = {
    val lines = (Source.fromFile(dataPath, "UTF-8").getLines() ++ List("")).toArray
    val xs = new ListBuffer[String]()
    val ys = new ListBuffer[String]()
    val indices = lines.zipWithIndex.filter(p => p._1.trim.isEmpty).map(p => p._2)
    var u = 0
    var v = 0
    for (i <- (0 until indices.length)) {
      v = indices(i)
      if (v > u) { // don't treat two consecutive empty lines
        val s = lines.slice(u, v)
        val tokens = s.map(line => {
          val parts = line.trim.split("""\t+""")
          if (parts.size < 2) {
            println(line + " at position " + u)
          }
          (parts(0), parts(1))
        })
        val x = tokens.map(_._1)
        xs.append(x.mkString(" "))
        val y = tokens.map(_._2)
        ys.append(y.mkString(" "))
      }
      u = v + 1
    }
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()    
    import spark.implicits._
    sc.parallelize(xs.zip(ys)).toDF("x", "y")
  }

  /**
    * Reads a test corpus in GED format (one column, *.tsv)
    * @param sc
    * @param dataPath
    * @return a data frame
    */
  def readTestDataGED(sc: SparkContext, dataPath: String): DataFrame = {
    val lines = (Source.fromFile(dataPath, "UTF-8").getLines() ++ List("")).toArray
    val xs = new ListBuffer[String]()
    val indices = lines.zipWithIndex.filter(p => p._1.trim.isEmpty).map(p => p._2)
    var u = 0
    var v = 0
    for (i <- (0 until indices.length)) {
      v = indices(i)
      if (v > u) { // don't treat two consecutive empty lines
        val s = lines.slice(u, v)
        val x = s.map(_.trim)
        xs.append(x.mkString(" "))
      }
      u = v + 1
    }
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()    
    import spark.implicits._
    sc.parallelize(xs).toDF("x").withColumn("y", lit("NA"))
  }  

  /**
    * Converts 2-col format to 4-col format of CoNLL-2003.
    *
    * @param inputPath
    * @param outputPath
    */
  def toCoNLL2003(inputPath: String, outputPath: String): Unit = {
    val lines = Source.fromFile(inputPath, "UTF-8").getLines().toArray
    val contents = lines.map { line =>
      val ts = line.trim.split("""\t+""")
      if (ts.size == 2) {
        ts(0).trim + " NA NA " + ts(1).trim
      } else {
        if (line.isEmpty()) "" else {
          println("ERR at line: " + line)
        }
      }
    }
    val fileWriter = new FileWriter(outputPath);
    for (str <- contents) {
      fileWriter.write(str + System.lineSeparator)
    }
    fileWriter.close()
  }
}

// toCoNLL2003("/Users/phuonglh/vlp/con/dat/med/med_ner_syll.tsv", "/Users/phuonglh/vlp/con/dat/med/syll.txt")