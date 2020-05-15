package vlp.vdr

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.types._

import scala.collection.mutable.ListBuffer


/**
  * phuonglh, 3/19/18, 11:17 AM
  *
  * A mapper transformer which maps known tokens/words in a predefined map.
  */
class Mapper(override val uid: String, val mappingPath: String) extends Transformer with DefaultParamsWritable {
  val mappings = IO.readMappings(mappingPath)
  var mappingsBr: Option[Broadcast[Map[String, String]]] = None
  
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final val predictionCol: Param[String] = new Param[String](this, "predictionCol", "prediction column name")
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  
  setDefault(inputCol -> "x", predictionCol -> "y", outputCol -> "z")
  
  def this(mappingPath: String) = this(Identifiable.randomUID("mapper"), mappingPath)

  private def createTransformFunc: (String, Seq[String]) => Seq[String] = {(x: String, ys: Seq[String]) => 
    if (mappingsBr.isEmpty) {
      val sparkContext = SparkSession.getActiveSession.get.sparkContext
      mappingsBr = Some(sparkContext.broadcast(mappings))
    }
    val broadcastMappings = mappingsBr.get
    ys.map { y =>
      var z = y
      broadcastMappings.value.keys.foreach { u =>
        // search for all occurrences of u in x
        val v = broadcastMappings.value(u)
        val indices = new ListBuffer[Int]
        var lastIndex = 0
        while (lastIndex != -1) {
          lastIndex = x.indexOf(u, lastIndex)
          if (lastIndex >= 0) {
            indices += lastIndex
            lastIndex += 1
          }
        }
        indices.foreach { p =>
          z = z.replaceAll(z.substring(p, p + v.size), v)
        }
      }
      z
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val transformUDF = udf(this.createTransformFunc, new ArrayType(StringType, false))
    dataset.withColumn($(outputCol), transformUDF(dataset($(inputCol)), dataset($(predictionCol))))
  }

  override def copy(extra: ParamMap) = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputFields = schema.fields :+
      StructField($(outputCol), new ArrayType(StringType, false), nullable = false)
    StructType(outputFields)
  }
}
