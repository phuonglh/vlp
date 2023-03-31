package vlp.woz.act

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataType, ArrayType, DoubleType}


/**
  * Select indices from a probability vector using a threshold value.
  * The indices are labels, so we convert it to Double type for evaluation 
  * with Spark API.
  * 
  * phuonglh@gmail.com
  */
class ThresholdSelector(val uid: String, val threshold: Float = 0.5f) extends 
  UnaryTransformer[Seq[Float], Seq[Double], ThresholdSelector] with DefaultParamsWritable {

  def this(epsilon: Float) = this(Identifiable.randomUID("thresholdSelector"), epsilon)
  def this() = this(0.5f)

  override protected def createTransformFunc: Seq[Float] => Seq[Double] = {
    def f(xs: Seq[Float]): Seq[Double] = {
      val pairs = xs.zipWithIndex
      // filter candidates above the given threshold
      val candidates = pairs.filter(_._1 >= threshold) 
      // if there is no value >= threshold, then select the best result
      if (candidates.isEmpty) pairs.sortBy(-_._1).take(1).map(_._2.toDouble) else candidates.map(_._2.toDouble)
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(DoubleType, false)
}

object ThresholdSelector extends DefaultParamsReadable[ThresholdSelector] {
  override def load(path: String): ThresholdSelector = super.load(path)
}