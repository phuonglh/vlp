package vlp.vdg

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{DataType, StringType}

/**
  * phuonglh, 10/13/18, 10:45
  *
  * Diacritic remover. Convert characters to lowercase.
  */
class DiacriticRemover(val uid: String) extends UnaryTransformer[String, String, DiacriticRemover] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("remover"))
  }

  override protected def createTransformFunc: String => String = {
    def f(x: String): String = {
      DiacriticRemover.run(x.toLowerCase)
    }
    
    f(_)
  }

  override protected def outputDataType: DataType = StringType
}

object DiacriticRemover extends DefaultParamsReadable[DiacriticRemover] {
  val map = new VieMap()

  /**
    * Remove diacritics of a string.
    * @param x accented string
    * @return non-accented string
    */
  def run(x: String): String = {
    x.map(c => map.getOrDefault(c, c)).mkString
  }

  override def load(path: String): DiacriticRemover = super.load(path)
}