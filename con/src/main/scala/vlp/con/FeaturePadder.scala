package vlp.con

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataType, ArrayType, FloatType}


/**
  * A feature padder.
  *
  * phuonglh@gmail.com
  */
class FeaturePadder(val uid: String, maxSequenceLength: Int, paddingValue: Float) 
  extends UnaryTransformer[Seq[Float], Seq[Float], FeaturePadder] with DefaultParamsWritable {

  var maxSeqLen: Int = -1
  var pad = 0f

  def this(maxSequenceLength: Int, paddingValue: Float) = {
    this(Identifiable.randomUID("padder"), maxSequenceLength, paddingValue)
    this.maxSeqLen = maxSequenceLength
    this.pad = paddingValue
  }

  override protected def createTransformFunc: Seq[Float] => Seq[Float] = {
    def f(xs: Seq[Float]): Seq[Float] = {
      if (xs.size >= maxSeqLen) {
        xs.take(maxSeqLen)
      } else {
        val p = Array.fill[Float](maxSeqLen - xs.size)(pad)
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