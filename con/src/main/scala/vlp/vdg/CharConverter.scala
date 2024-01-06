package vlp.vdg

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

/**
  * phuonglh, 10/22/18, 21:11
  * <p>
  *   Convert numeric characters to "0", space character to ' ', 
  *   and all other simple characters (such as '_', '%', etc.) 
  *   to a special S character (for SYMBOL).
  */
class CharConverter(val uid: String) extends UnaryTransformer[Seq[String], Seq[String], CharConverter] with DefaultParamsWritable {
  
  def this() = this(Identifiable.randomUID("charConverter"))
  
  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    def f(xs: Seq[String]): Seq[String] = {
      xs.map(x => CharConverter.convert(x))
    }
    
    f(_)
  }

  override protected def outputDataType: DataType = new ArrayType(StringType, false)
}

object CharConverter extends DefaultParamsReadable[CharConverter] {
  val vieMap = new VieMap()
  
  def convert(token: String): String = {
    if (WordShape.shape(token) == "number") "0"
    else if (!vieMap.contains(token)) "S"
    else token
  }
  
  override def load(path: String): CharConverter = super.load(path)
}
