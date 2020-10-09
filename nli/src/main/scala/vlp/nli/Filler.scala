package vlp.nli

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataType
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

/**
  *
  * Fill all rows with a given constant. This is useful for creating token type indices for BERT model.
  * 
  * phuonglh@gmail.com
  */
class Filler(val uid: String, constant: Double) extends UnaryTransformer[Vector, Vector, Filler] with DefaultParamsWritable {

  def this(constant: Double) = this(Identifiable.randomUID("filler"), constant)

  override protected def createTransformFunc: Vector => Vector = {
    def f(x: Vector): Vector = {
      val values = Array.fill(x.size)(constant)
      Vectors.dense(values)
    }

    f(_)
  }

  override protected def outputDataType: DataType = VectorType
}

object Filler extends DefaultParamsReadable[Filler] {
  override def load(path: String): Filler = super.load(path)
}