package vlp.vdr

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StringType

/**
  * phuonglh, 11/2/17, 17:12
  * 
  * Accent remover which transforms a normal sentence into a non-accent one.
  */
class Remover(override val uid: String) extends UnaryTransformer[String, String, Remover] with DefaultParamsWritable {
  
  def this() = this(Identifiable.randomUID("remover"))
  
  override protected def createTransformFunc: (String) => String = {
    _.map(c => VieMap.diacritics.getOrElse(c, c))
  }

  override protected def outputDataType = StringType
}
