package vlp.woz.act

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataType, ArrayType, DoubleType}


/**
  * Select top-k indices from a probability vector using a threshold value.
  * The indices are labels, so we convert it to Double type for evaluation 
  * with Spark API.
  * 
  * phuonglh@gmail.com
  */
class TopKSelector(val uid: String, val k: Int, val threshold: Float = 0.1f) extends 
  UnaryTransformer[Seq[Float], Seq[Double], TopKSelector] with DefaultParamsWritable {

  def this(k: Int) = this(Identifiable.randomUID("topK"), k)
  def this(k: Int, threshold: Float) = this(Identifiable.randomUID("topK"), k, threshold)

  override protected def createTransformFunc: Seq[Float] => Seq[Double] = {
    def f(xs: Seq[Float]): Seq[Double] = {
      // sort by values in decreasing order
      val topK = xs.zipWithIndex.sortBy(-_._1).take(k)
      // filter candidates above the given threshold
      val candidates = topK.filter(_._1 >= threshold) 
      // return result
      if (candidates.isEmpty) topK.take(1).map(_._2.toDouble) else candidates.map(_._2.toDouble)
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(DoubleType, false)
}

object TopKSelector extends DefaultParamsReadable[TopKSelector] {
  override def load(path: String): TopKSelector = super.load(path)
}