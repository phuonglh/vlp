package vlp.vsc

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataType, ArrayType, StringType}


/**
  * A subtoken transformer which transforms a token sequence into a subtoken sequence:
    [John, loves, Mary] => [J, oh, n, l, ove, s, M, ar, y]. This utility is used to build a subtoken vocabulary.

  * phuonglh@gmail.com
  */
class SubtokenTransformer(val uid: String) 
  extends UnaryTransformer[Seq[String], Seq[String], SubtokenTransformer] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("subtok"))
  }

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    def f(xs: Seq[String]): Seq[String] = {
      xs.flatMap(SubtokenTransformer.s(_))
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(StringType, false)
}

object SubtokenTransformer extends DefaultParamsReadable[SubtokenTransformer] {
  def s(x: String): Seq[String] = {
    x.size match {
      case 0 => Seq.empty[String]
      case 1 => Seq(x, "$NA", "$NA")
      case 2 => Seq(x.take(1), "$NA", x.takeRight(1))
      case _ => Seq(x.take(1), x.substring(1, x.size-1), x.takeRight(1))
    }
  }

  override def load(path: String): SubtokenTransformer = super.load(path)
}