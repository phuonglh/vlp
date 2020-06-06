package vlp.vdg

import vlp.tok.WordShape
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

class TokenConverter(val uid: String) extends UnaryTransformer[Seq[String], Seq[String], TokenConverter] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("tokenConverter"))

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    def f(xs: Seq[String]): Seq[String] = {
      xs.map(x => TokenConverter.convert(x))
    }

    f(_)
  }

  override protected def outputDataType: DataType = new ArrayType(StringType, false)
}

object TokenConverter extends DefaultParamsReadable[CharConverter] {

  def convert(token: String): String = {
    if (WordShape.shape(token) == "number") "0"
    else if (WordShape.shape(token) == "punctuation") "S"
    else if (token == " ") " "
    else token
  }

  override def load(path: String): CharConverter = super.load(path)
}
