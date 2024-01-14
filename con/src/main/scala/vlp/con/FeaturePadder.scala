package vlp.con

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, FloatType}


/**
  * A feature padder.
  *
  * phuonglh@gmail.com
  */
class FeaturePadder(val uid: String, maxSequenceLength: Int, paddingValue: Float)
  extends UnaryTransformer[Seq[Float], Seq[Float], FeaturePadder] with DefaultParamsWritable {

  def this(maxSequenceLength: Int, paddingValue: Float) = {
    this(Identifiable.randomUID("padder"), maxSequenceLength, paddingValue)
  }

  override protected def createTransformFunc: Seq[Float] => Seq[Float] = {
    def f(xs: Seq[Float]): Seq[Float] = {
      if (xs.size >= maxSequenceLength) {
        xs.take(maxSequenceLength)
      } else {
        val p = Array.fill[Float](maxSequenceLength - xs.size)(paddingValue)
        xs ++ p
      }
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(FloatType, false)
}

object FeaturePadder extends DefaultParamsReadable[FeaturePadder] {
  override def load(path: String): FeaturePadder = super.load(path)
}