package vlp.dep

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}


/**
  A character sequencer which transforms a sequence of tokens into a sequence of 13-character subsequences.
    [John, loves, Mary] => [J, o, h, n,... l, o, v, e, s,... M, a, r, y,...]. The dots are padding characters
  until maxSeqLen. The special padding symbol is "$P".

  <p/>
  * phuonglh@gmail.com
  */
class CharacterSequencer(val uid: String, val maxSeqLen: Int = 13)
  extends UnaryTransformer[Seq[String], Seq[String], CharacterSequencer] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID(" charSeq"))
  }

  def this(maxSeqLen: Int) = {
    this(Identifiable.randomUID(" charSeq"), maxSeqLen)
  }

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    def f(xs: Seq[String]): Seq[String] = {
      xs.flatMap(CharacterSequencer.s(_, maxSeqLen))
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(StringType, false)
}

object CharacterSequencer extends DefaultParamsReadable[CharacterSequencer] {
  def s(x: String, maxSeqLen: Int): Seq[String] = {
    x.length match {
      case 0 => Seq.empty[String]
      case _ =>
        val y = x.map(_.toString)
        // pad or truncate
        if (x.length < maxSeqLen) y ++ Seq.fill(maxSeqLen - x.length)("$P") else y.take(maxSeqLen)
    }
  }

  override def load(path: String): CharacterSequencer = super.load(path)
}