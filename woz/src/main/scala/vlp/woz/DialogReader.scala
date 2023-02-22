package vlp.woz

import org.apache.spark.SparkConf
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._

import org.apache.spark.sql.catalyst.ScalaReflection

import org.apache.log4j.{Logger, Level}
import org.slf4j.LoggerFactory
import vlp.woz.act.DialogActReader

/**
  * phuonglh, 2023
  * 
  */
object DialogReader {
  val logger = LoggerFactory.getLogger(getClass.getName)

  def readDialogs(spark: SparkSession, split: String): DataFrame = {
    import spark.implicits._
    // SlotValues are temporarily set to empty. TODO: give a complete description for this struct type. 
    // We use a ScalaReflection hack to overcome a schema error of empty SlotValues type:
    val scalaSchema = ScalaReflection.schemaFor[SlotValues].dataType.asInstanceOf[StructType]
    val path = s"dat/woz/data/MultiWOZ_2.2/${split}"
    // read the whole directory of the split (train/dev/test)
    val df = spark.read.option("multiline", "true").json(path)
    val ef = df.as[Dialog]
    // extract dialogId, turnId and utterance from the dialog
    val ff = ef.flatMap(d => d.turns.map(t => (d.dialogue_id, t.turn_id, t.utterance)))
    ff.toDF("dialogId", "turnId", "utterance")
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[2]")
    val spark = SparkSession.builder.config(conf).getOrCreate()
        
    val df = readDialogs(spark, "dev")
    df.show(false)
    println(s"Number of turns = ${df.count}")

    val as = DialogActReader.readAll()
    import spark.implicits._
    val af = spark.sparkContext.parallelize(as).toDF("dialogId", "turnId", "actNames")
    af.show(false)
    println(s"Number of turns in the act dataset = ${af.count}")
    af.printSchema()

    // inner join of two data frames using dialogId and turnId
    // then sort the resulting data frame by dialogId and turnId
    val ff = df.as("df").join(af, df("dialogId") === af("dialogId") && df("turnId") === af("turnId"), "inner")
      .select("df.*", "actNames") // select columns from df to avoid duplicates of column names
      .sort(col("dialogId"), col("turnId").cast("int")) // need to cast turnId to int before sorting
    ff.show(50, false)
    println(ff.count)

    spark.close()
  }
}
