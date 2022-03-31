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

import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

/**
 * phuonglh, July 2020
 * 
 * Stacks two similar length vectors u = [u_1,...,u_n] and v = [v_1,..., v_n] into a long 
 * [u_1,..., u_n, v_1,.... v_n]
 * 
 */
class VectorStacker(override val uid: String) extends Transformer with HasInputCols with HasOutputCol
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("vecStacker"))

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  private def createTransformFunc: (Vector, Vector) => Vector = {(u: Vector, v: Vector) => 
    Vectors.dense(u.toArray ++ v.toArray)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val schema = dataset.schema
    // val transformUDF = udf(this.createTransformFunc, VectorType)
    val transformUDF = udf(this.createTransformFunc)
    dataset.withColumn($(outputCol), transformUDF(dataset($(inputCols)(0)), dataset($(inputCols)(1))))
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputColNames = $(inputCols)
    val outputColName = $(outputCol)
    val incorrectColumns = inputColNames.flatMap { name =>
      schema(name).dataType match {
        case VectorType => None
        case other => Some(s"Data type ${other.catalogString} of column $name is not supported.")
      }
    }
    if (incorrectColumns.nonEmpty) {
      throw new IllegalArgumentException(incorrectColumns.mkString("\n"))
    }
    if (schema.fieldNames.contains(outputColName)) {
      throw new IllegalArgumentException(s"Output column $outputColName already exists.")
    }
    StructType(schema.fields :+ new StructField(outputColName, VectorType, true))
  }

  override def copy(extra: ParamMap): VectorStacker = defaultCopy(extra)  
}

object VectorStacker extends DefaultParamsReadable[VectorStacker] {
  override def load(path: String): VectorStacker = super.load(path)
}