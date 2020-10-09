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
  * Fill all rows with position indices. This is useful for creating position indices for BERT model.
  * 
  * phuonglh@gmail.com
  */
class PositionFiller(val uid: String, maxSeqLen: Int) extends UnaryTransformer[Vector, Vector, PositionFiller] with DefaultParamsWritable {

  def this(maxSeqLen: Int) = this(Identifiable.randomUID("posFiller"), maxSeqLen)

  override protected def createTransformFunc: Vector => Vector = {
    def f(x: Vector): Vector = {
      val values = (0 until maxSeqLen).toArray.map(_.toDouble)
      Vectors.dense(values)
    }

    f(_)
  }

  override protected def outputDataType: DataType = VectorType
}

object PositionFiller extends DefaultParamsReadable[PositionFiller] {
  override def load(path: String): PositionFiller = super.load(path)
}