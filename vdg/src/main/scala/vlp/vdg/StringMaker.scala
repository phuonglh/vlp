package vlp.vdg

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{DataType, StringType}

/**
  * phuonglh, 10/24/18, 21:42
  * 
  * Concatenate a sequence of tokens into a string with the default empty separator.
  */
class StringMaker(val uid: String) extends UnaryTransformer[Seq[String], String, StringMaker] with DefaultParamsWritable {
  
  def this() = this(Identifiable.randomUID("strMaker"))
  
  override protected def createTransformFunc: Seq[String] => String = {
    _.mkString
  }

  override protected def outputDataType: DataType = StringType
}

object StringMaker extends DefaultParamsReadable[StringMaker] {
  override def load(path: String): StringMaker = super.load(path)
}
