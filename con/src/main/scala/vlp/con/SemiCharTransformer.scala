package vlp.con

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataType, ArrayType, StringType}


/**
  * A semi-character transformer which transforms a token sequence into a sub-token sequence:
    [John, loves, Mary] => [J, oh, n, l, ove, s, M, ar, y]. This utility is used to build sub-token 
    vocabulary.

  * phuonglh@gmail.com
  */
class SemiCharTransformer(val uid: String) 
  extends UnaryTransformer[Seq[String], Seq[String], SemiCharTransformer] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("semiChar"))
  }

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    def f(xs: Seq[String]): Seq[String] = {
      xs.flatMap(SemiCharTransformer.s(_))
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(StringType, false)
}

object SemiCharTransformer extends DefaultParamsReadable[SemiCharTransformer] {
    def s(x: String): Seq[String] = {
      x.size match {
        case 0 => Seq.empty[String]
        case 1 => Seq(x, "$NA", "$NA")
        case 2 => Seq(x.take(1), "$NA", x.takeRight(1))
        case _ => Seq(x.take(1), x.substring(1, x.size-1), x.takeRight(1))
      }
    }

  override def load(path: String): SemiCharTransformer = super.load(path)
}