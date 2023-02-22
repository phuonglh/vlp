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

  /**
    * Reads the dialog act data frames. There will be 3 data frames: train/dev/test.
    * Each data frame has 4 columns: (dialogId, turnId, utterance, acts)
    * 
    * @param spark
    * @param save
    * @return a sequence of 3 data frames corresponding to train/dev/test split of the WoZ corpus.
    */
  def readDialogActs(spark: SparkSession, save: Boolean = false): Seq[DataFrame] = {
    val splits = Seq("train", "dev", "test")
    import spark.implicits._
    // read all dialog acts
    val as = DialogActReader.readAll()
    val af = spark.sparkContext.parallelize(as).toDF("dialogId", "turnId", "actNames")
    splits.map { split => 
      val df = readDialogs(spark, split)
      // inner join of two data frames using dialogId and turnId
      // then sort the resulting data frame by dialogId and turnId
      val ff = df.as("df").join(af, df("dialogId") === af("dialogId") && df("turnId") === af("turnId"), "inner")
        .select("df.*", "actNames") // select columns from df to avoid duplicates of column names
        .sort(col("dialogId"), col("turnId").cast("int")) // need to cast turnId to int before sorting
        // save the df to json
        if (save)
          ff.repartition(1).write.option("header", "true").option("delimiter", "\t").json(s"dat/woz/act/$split")
      ff
    }
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[2]")
    val spark = SparkSession.builder.config(conf).getOrCreate()
    val dfs = readDialogActs(spark, true)
    println(s"#(trainingSamples) = ${dfs(0).count}")
    println(s"     #(devSamples) = ${dfs(1).count}")
    println(s"    #(testSamples) = ${dfs(2).count}")
    dfs(2).show()
    dfs(2).printSchema()

    spark.close()
  }
}
