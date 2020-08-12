package vlp.ner

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}
import vlp.tok.WordShape

/**
  * phuonglh@gmail.com
  *
  * Transform a sequence of words to a sequence of shapes. 
  */
class WordShaper(override val uid: String) extends UnaryTransformer[Seq[String], Seq[String], WordShaper] with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("shaper"))

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    _.map(x => {
      val sh = WordShape.shape(x)
      if (sh.isEmpty()) "word" else sh
    })
  }

  override protected def outputDataType: DataType = ArrayType(StringType, false)
}

object WordShaper extends DefaultParamsReadable[WordShaper] {
  override def load(path: String): WordShaper = super.load(path)
}
