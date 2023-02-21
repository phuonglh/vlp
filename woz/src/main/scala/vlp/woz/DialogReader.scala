package vlp.woz

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._

import org.apache.spark.sql.catalyst.ScalaReflection

import org.apache.log4j.{Logger, Level}
import org.slf4j.LoggerFactory

/**
  * phuonglh, 2023
  * 
  */
object DialogReader {
  val logger = LoggerFactory.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[2]")
    val spark = SparkSession.builder.config(conf).getOrCreate()
    import spark.implicits._

    val path = "dat/woz/data/MultiWOZ_2.2/dev/dialogues_002.json"
    val domain = if (args.size == 0) "train" else args(0)
    // SlotValues are temporarily set to empty. TODO: give a complete description for this struct type. 
    // We use a ScalaReflection hack to overcome a schema error of empty SlotValues type:
    val scalaSchema = ScalaReflection.schemaFor[SlotValues].dataType.asInstanceOf[StructType]

    val df = spark.read.option("multiline", "true").json(path)
    val ef = df.as[Dialog]
    // extract dialogId, turnId and utterance from the dialog
    val ff = ef.flatMap(d => d.turns.map(t => (d.dialogue_id, t.turn_id, t.utterance)))
    val gf = ff.toDF("dialogId", "turnId", "utterance")

    gf.show(false)
    println(gf.count())

    spark.close()
  }
}
