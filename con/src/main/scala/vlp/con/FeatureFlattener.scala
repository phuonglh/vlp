package vlp.con

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{DataType, ArrayType, FloatType}


/**
  * A feature flattener.
  *
  * phuonglh@gmail.com
  */
class FeatureFlattener(val uid: String) extends UnaryTransformer[Seq[Seq[Float]], Seq[Float], FeatureFlattener] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("flattener"))
  }

  override protected def createTransformFunc: Seq[Seq[Float]] => Seq[Float] = {
    _.flatten
  }

  override protected def outputDataType: DataType = ArrayType(FloatType, false)
}

object FeatureFlattener extends DefaultParamsReadable[FeatureFlattener] {
  override def load(path: String): FeatureFlattener = super.load(path)
}