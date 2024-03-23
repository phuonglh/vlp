package vlp.con

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.DataType


/**
  * A sequence vectorizer transforms a sequence of tokens into a sequence of indices
  * using a dictionary. This transformer pads or truncate long sentence to a given `maxSequenceLength`.
  * If the dictionary does not contain a token, it returns one (1, since BigDL uses 1-based index; this prevent errors
  * -- the target sequence never contains 0.).
  *
  * phuonglh@gmail.com
  */
class Sequencer(val uid: String, val dictionary: Map[String, Int], val maxSequenceLength: Int, val padding: Float) 
  extends UnaryTransformer[Seq[String], Vector, Sequencer] with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Int]]] = None
  var maxSeqLen: Int = -1
  var pad: Float = -1f

  def this(dictionary: Map[String, Int], maxSequenceLength: Int, padding: Float) = {
    this(Identifiable.randomUID("seq"), dictionary, maxSequenceLength, padding)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
    this.maxSeqLen = maxSequenceLength
    this.pad = padding
  }

  override protected def createTransformFunc: Seq[String] => Vector = {
    def f(xs: Seq[String]): Vector = {
      val a = xs.map(x => dictionaryBr.get.value.getOrElse(x, 1).toDouble).toArray
      // truncate or pad
      if (xs.size >= maxSeqLen) {
        Vectors.dense(a.take(maxSeqLen))
      } else {
        val b = a ++ Array.fill[Double](maxSeqLen - xs.size)(pad)
        Vectors.dense(b)
      }
    }

    f(_)
  }

  override protected def outputDataType: DataType = VectorType
}

object Sequencer extends DefaultParamsReadable[Sequencer] {
  override def load(path: String): Sequencer = super.load(path)
}