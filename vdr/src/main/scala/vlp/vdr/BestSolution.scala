package vlp.vdr

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

/**
  * phuonglh, 3/20/18, 3:22 PM
  */
class BestSolution(override val uid: String) extends Transformer with DefaultParamsWritable {
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  final val column: IntParam = new IntParam(this, "column", "selected column", ParamValidators.gtEq(0))

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setColumn(value: Int): this.type = set(column, value)

  setDefault(inputCol -> "y", outputCol -> "prediction", column -> 0)

  def this() = this(Identifiable.randomUID("bestSol"))

  protected def createTransformFunc: (Seq[String]) => String = { (input: Seq[String]) =>
    input($(column))
  }
  
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val transformUDF = udf(this.createTransformFunc, StringType)
    dataset.withColumn($(outputCol), transformUDF(dataset($(inputCol))))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputFields = schema.fields :+
      StructField($(outputCol), StringType, nullable = false)
    StructType(outputFields)
  }
}
