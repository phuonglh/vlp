package vlp.tpm

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}
import vlp.tok.WordShape

/**
  * A transformer to pre-process tokens.
  *
  * phuonglh
  * @param uid
  */
class Converter(override val uid: String) extends UnaryTransformer[Seq[String], Seq[String], Converter] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("converter"))

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    _.map(token => WordShape.shape(token) match {
      case "email" => "<email>"
      case "url" => "<url>"
      case "date" => "<date>"
      case "time" => "<time>"
      case "percentage" => "<percent>"
      case "number" => "<number>"
      case _ => token.toLowerCase
    })
  }

  override protected def outputDataType: DataType = ArrayType(StringType, false)
}

object Converter extends DefaultParamsReadable[Converter] {
  override def load(path: String): Converter = super.load(path)
}
