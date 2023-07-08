package vlp.med

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types.{DataType, DoubleType}


/**
  * An OneCold transforms an one-hot vector into an integer index (start from 0).
  *
  * phuonglh@gmail.com
  */
class OneCold(val uid: String) extends UnaryTransformer[Vector, Double, OneCold] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("oneCold"))
  }

  override protected def createTransformFunc: Vector => Double = {
    def f(x: Vector): Double = x.toSparse.indices(0).toDouble
    f(_)
  }

  override protected def outputDataType: DataType = DoubleType
}

object OneCold extends DefaultParamsReadable[OneCold] {
  override def load(path: String): OneCold = super.load(path)
}