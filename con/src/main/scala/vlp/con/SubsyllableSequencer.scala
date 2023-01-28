package vlp.con

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.DataType


/**
  * A subsyllable sequencer which transforms a sequence of tokens into a sequence 
  * of subsyllable index [b, i, e] for each token in the sequence. A dictionary which is 
  * obtained by [[SubsyllableTransformer]] is used to create indices. If there are n tokens then 
    this sequencer produces a vector of (3*n) elements.
  *
  * phuonglh@gmail.com
  */
class SubsyllableSequencer(val uid: String, val dictionary: Map[String, Int], maxSequenceLength: Int, padding: Float) 
  extends UnaryTransformer[Seq[String], Vector, SubsyllableSequencer] with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Int]]] = None
  var maxSeqLen: Int = -1
  var pad: Float = -1f

  def this(dictionary: Map[String, Int], maxSequenceLength: Int, padding: Float) = {
    this(Identifiable.randomUID("subsyllSeq"), dictionary, maxSequenceLength, padding)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
    this.maxSeqLen = maxSequenceLength
    this.pad = padding
  }

  override protected def createTransformFunc: Seq[String] => Vector = {
    def f(xs: Seq[String]): Vector = {
      val a = xs.flatMap{x => 
        val ts = SubsyllableTransformer.s(x)
        ts.map(t => dictionaryBr.get.value.getOrElse(t, 0).toDouble)
      }.toArray
      // truncate or pad
      if (xs.size >= maxSeqLen) {
        Vectors.dense(a.take(3*maxSeqLen))
      } else {
        val b = a ++ Array.fill[Double](3*(maxSeqLen - xs.size))(pad)
        Vectors.dense(b)
      }
    }

    f(_)
  }

  override protected def outputDataType: DataType = VectorType
}

object SubsyllableSequencer extends DefaultParamsReadable[SubsyllableSequencer] {
  override def load(path: String): SubsyllableSequencer = super.load(path)
}