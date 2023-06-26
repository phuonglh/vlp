package vlp.woz

import org.apache.spark.SparkConf
import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.catalyst.ScalaReflection

import vlp.woz.act.DialogActReader

/**
  * phuonglh, 2023
  * 
  */
object DialogReader {

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
        if (save) ff.repartition(1).write.json(s"dat/woz/act/$split")
      ff
    }
  }

  /**
    * For each turn in a df (read by the [[readDialogActs()]] method), concat the act history at previous 3 turns.
    * First, we use the `concat_ws()` function to flatten the `actNames` column (say, turn an array `[Hotel-Inform, Hotel-Select]`
    * to a space-delimited string `Hotel-Inform Hotel-Select`). 
    * 
    * We need to define a window on the frame which is partitioned by `dialogId`; the rows in each 
    * partition is ordered by integer-valued `turnId`. 
    * Then we use the `lag()` function repeatedly to extract acts(-3), acts(-2), acts(-1) values of the current act.
    * Finally, we concatenate 3 acts(-i) columns together.
    * @param spark
    * @param df
    */
  def concatDialogActs(spark: SparkSession, df: DataFrame): DataFrame = {
    val df1 = df.withColumn("acts", concat_ws(" ", col("actNames")))
    // define a window
    val window = Window.partitionBy("dialogId").orderBy(col("turnId").cast("int"))
    // repeatedly add 3 columns for previous acts
    val df2 = df1.withColumn("acts(-3)", lag("actNames", 3).over(window))
    val df3 = df2.withColumn("acts(-2)", lag("actNames", 2).over(window))
    val df4 = df3.withColumn("acts(-1)", lag("actNames", 1).over(window))
    // concatenate 3 columns
    df4.withColumn("prevActs", concat_ws(" ", col("acts(-3)"), col("acts(-2)"), col("acts(-1)")))
  }

  /**
    * Reads WOZ dialog acts and writes splits out.
    *
    * @param spark
    * @param save
    * @return 3 dataframes
    */
  def readDialogActsWOZ(spark: SparkSession, save: Boolean = false): Seq[DataFrame] = {
    val dfs = readDialogActs(spark, save)
    println(s"#(trainingSamples) = ${dfs(0).count}")
    println(s"     #(devSamples) = ${dfs(1).count}")
    println(s"    #(testSamples) = ${dfs(2).count}")
    dfs(2).show()
    dfs(2).printSchema()
    dfs
  }

  /**
    * Reads FPT dialog acts and writes out
    *
    * @param spark
    * @param path
    * @param save
    */
  def readDialogActsFPT(spark: SparkSession, path: String, save: Boolean = false): DataFrame = {
    val df = spark.read.format("com.databricks.spark.xml").option("rootTag", "TEI.DIALOG").option("rowTag", "utterance").load(path)
    df.printSchema()
    // extract the samples: (utteranceId, "utterance", Seq[communicativeFunction])
    val rdd = df.rdd.map { row => Row(
        row.getAs[Row]("txt").getAs[Long]("_id").toString,
        row.getAs[Row]("txt").getAs[String]("_VALUE"),
        // use flatMap to flatten communicativeFunctions elements
        row.getAs[Seq[Row]]("act").flatMap(_.getAs[Seq[String]]("communicativeFunction"))
      )
    }
    val schema = StructType(Seq(
      StructField("turnId", StringType, true),
      StructField("utterance", StringType, true),
      StructField("actNames", ArrayType(StringType, true), true)
    ))
    val ef = spark.createDataFrame(rdd, schema)
    if (save) ef.repartition(1).write.json(s"dat/vie/act/")
    ef
  }

  def readDialogStatesWOZ(spark: SparkSession, split: String): DataFrame = {
    import spark.implicits._
    // We use a ScalaReflection hack to overcome a schema error of empty SlotValues type:
    val scalaSchema = ScalaReflection.schemaFor[SlotValues].dataType.asInstanceOf[StructType]
    val path = s"dat/woz/data/MultiWOZ_2.2/${split}"
    // read the whole directory of the split (train/dev/test)
    val df = spark.read.option("multiline", "true").json(path)
    df.printSchema
    val ef = df.as[Dialog]
    // extract dialogId, turnId, utterance, and states
    val ff = ef.flatMap { d => 
      val services = d.services.toSet
      d.turns.map { t => 
        val states = t.frames.filter(f => services.contains(f.service)).map(_.state)
        (d.dialogue_id, t.turn_id, t.utterance, states)
      }
    }
    ff.toDF("dialogId", "turnId", "utterance", "states")
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[2]")
    val spark = SparkSession.builder.config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    // val dfs = readDialogActsWOZ(spark, true)
    // val df = readDialogActsFPT(spark, "dat/fpt/", true)
    val df = readDialogStatesWOZ(spark, "test")
    df.show()
    // println(df.count())
    // df.select("states").show(false)
    spark.close()
  }
}
