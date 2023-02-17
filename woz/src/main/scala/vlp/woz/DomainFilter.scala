package vlp.woz

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._

import org.apache.spark.sql.catalyst.ScalaReflection


object DomainFilter {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[2]")
    val spark = SparkSession.builder.config(conf).getOrCreate()
    import spark.implicits._

    val path = "dat/woz/data/MultiWOZ_2.2/dev/dialogues_001.json"
    val domain = if (args.size == 0) "train" else args(0)
    // SlotValues are temporarily set to empty. TODO: give a complete description for this struct type. 
    // We use a ScalaReflection hack to overcome a schema error.
    val scalaSchema = ScalaReflection.schemaFor[SlotValues].dataType.asInstanceOf[StructType]

    val df = spark.read.option("multiline", "true").json(path)
    val ef = df.filter(col("dialogue_id").startsWith("SNG"))
    val ff = ef.as[Dialogue]    
    val gf = ff.filter(exists(col("services"), _.contains(domain)))
    // flat map all the turns of the dialogues using the `turns` field
    val hf = gf.flatMap(_.turns)
    // flat map all the frames, keep `speaker` and `utterance` along the way
    // since the dialogues are in a single domain, we remove out-of-domain frames because they are all empty
    // it is always better to filter before mapping
    val kf = hf.flatMap(turn => turn.frames.filter(_.service == domain).map(frame => 
      (turn.speaker, turn.utterance, frame.state.active_intent, frame.state.requested_slots))
    ).toDF("speaker", "utterance", "activeIntent", "requestedSlots")

    kf.show()
    println(kf.count())

    spark.close()
  }
}
