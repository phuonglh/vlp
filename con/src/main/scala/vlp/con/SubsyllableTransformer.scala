package vlp.con

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataType, ArrayType, StringType}


/**
  * A subsyllable transformer which transforms a syllable sequence into a subsyllable sequence:
    [John, loves, Mary] => [J, oh, n, l, ove, s, M, ar, y]. This utility is used to build sub-syllable 
    vocabulary.

  * phuonglh@gmail.com
  */
class SubsyllableTransformer(val uid: String) 
  extends UnaryTransformer[Seq[String], Seq[String], SubsyllableTransformer] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("subsyll"))
  }

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    def f(xs: Seq[String]): Seq[String] = {
      xs.flatMap(SubsyllableTransformer.s(_))
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(StringType, false)
}

object SubsyllableTransformer extends DefaultParamsReadable[SubsyllableTransformer] {
  def s(x: String): Seq[String] = {
    x.size match {
      case 0 => Seq.empty[String]
      case 1 => Seq(x, "$NA", "$NA")
      case 2 => Seq(x.take(1), "$NA", x.takeRight(1))
      case _ => Seq(x.take(1), x.substring(1, x.size-1), x.takeRight(1))
    }
  }

  override def load(path: String): SubsyllableTransformer = super.load(path)
}