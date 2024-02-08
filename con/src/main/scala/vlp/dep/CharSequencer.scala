package vlp.dep

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}


/**
  A character sequencer which transforms a sequence of tokens into a sequence of 13-character subsequences.
    [John, loves, Mary] => [J, o, h, n,... l, o, v, e, s,... M, a, r, y,...]. The dots are padding characters
  until maxSeqLen. The special padding symbol is "$P".

  <p/>
  * phuonglh@gmail.com
  */
class CharSequencer(val uid: String)
  extends UnaryTransformer[Seq[String], Seq[String], CharSequencer] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID(" charSeq"))
  }

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    def f(xs: Seq[String]): Seq[String] = {
      xs.flatMap(CharSequencer.s(_, 13)) // FIX at 13
    }
    f
  }

  override protected def outputDataType: DataType = ArrayType(StringType, containsNull = false)

  override def copy(extra: ParamMap): CharSequencer = defaultCopy(extra)
}

object CharSequencer extends DefaultParamsReadable[CharSequencer] {
  def s(x: String, maxSeqLen: Int): Seq[String] = {
    x.length match {
      case 0 => Seq.empty[String]
      case _ =>
        val y = x.map(_.toString)
        // pad or truncate
        if (x.length < maxSeqLen) y ++ Seq.fill(maxSeqLen - x.length)("$P") else y.take(maxSeqLen)
    }
  }

  override def load(path: String): CharSequencer = super.load(path)
}