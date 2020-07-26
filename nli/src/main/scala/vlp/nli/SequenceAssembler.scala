package vlp.nli

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.adapter.HasOutputCol
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.DefaultParamsReadable
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{udf, col}
import org.apache.spark.SparkException
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.attribute.AttributeGroup

/**
 * phuonglh, July 2020
 */
class SequenceAssembler(override val uid: String) extends Transformer with HasInputCols with HasOutputCol
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("seqAssembler"))

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def createTransformFunc: (Seq[String], Seq[String]) => Seq[String] = {(xs: Seq[String], ys: Seq[String]) => 
    xs ++ ys
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val schema = dataset.schema
    val transformUDF = udf(this.createTransformFunc, new ArrayType(StringType, false))
    dataset.withColumn($(outputCol), transformUDF(dataset($(inputCols)(0)), dataset($(inputCols)(1))))
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputColNames = $(inputCols)
    val outputColName = $(outputCol)
    val incorrectColumns = inputColNames.flatMap { name =>
      schema(name).dataType match {
        case _: ArrayType => None
        case other => Some(s"Data type ${other.catalogString} of column $name is not supported.")
      }
    }
    if (incorrectColumns.nonEmpty) {
      throw new IllegalArgumentException(incorrectColumns.mkString("\n"))
    }
    if (schema.fieldNames.contains(outputColName)) {
      throw new IllegalArgumentException(s"Output column $outputColName already exists.")
    }
    StructType(schema.fields :+ new StructField(outputColName, new ArrayType(StringType, false), true))
  }

  override def copy(extra: ParamMap): SequenceAssembler = defaultCopy(extra)  
}

object SequenceAssembler extends DefaultParamsReadable[SequenceAssembler] {
  override def load(path: String): SequenceAssembler = super.load(path)
}