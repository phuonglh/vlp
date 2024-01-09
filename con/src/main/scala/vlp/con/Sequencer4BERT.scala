package vlp.con

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.DataType


/**
  * A sequence vectorizer transforms a sequence of tokens into 4 sequences of indices for use 
  * in a BERT model. This transformer pads or truncate long sentence to a given `maxSequenceLength`.
  * A dictionary is needed to convert tokens into indices. The output is a vector if length 4 x 'maxSequenceLength".
  * See the BERT paper for detail.
 *
  * phuonglh@gmail.com
  */
class Sequencer4BERT(val uid: String, val dictionary: Map[String, Int], val maxSequenceLength: Int, val padding: Float)
  extends UnaryTransformer[Seq[String], Vector, Sequencer4BERT] with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Int]]] = None
  var maxSeqLen: Int = -1
  var pad: Float = -1f

  def this(dictionary: Map[String, Int], maxSequenceLength: Int, padding: Float) = {
    this(Identifiable.randomUID("seq4BERT"), dictionary, maxSequenceLength, padding)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
    this.maxSeqLen = maxSequenceLength
    this.pad = padding
  }

  override protected def createTransformFunc: Seq[String] => Vector = {
    def f(xs: Seq[String]): Vector = {
      val n = xs.size
      // token ids
      val tokens = xs.map(x => dictionaryBr.get.value.getOrElse(x, 0).toDouble).toArray
      // token type, all are 0 (0 for sentence A, 1 for sentence B -- here we have only one sentence)
      val types = Array.fill[Double](n)(0)
      // positions, start from 0 until xs.size
      val positions = Array.fill[Double](n)(0)
      for (j <- 0 until n)
        positions(j) = j
      // attention masks with indices in [0, 1]
      // It's a mask to be used if the input sequence length is smaller than maxSeqLen
      val masks = Array.fill[Double](n)(1)

      // truncate or pad
      if (n >= maxSeqLen) {
        Vectors.dense(tokens.take(maxSeqLen) ++ types.take(maxSeqLen) ++ positions.take(maxSeqLen) ++ masks.take(maxSeqLen))
      } else {
        val a = tokens    ++ Array.fill[Double](maxSeqLen - xs.size)(pad)
        val b = types     ++ Array.fill[Double](maxSeqLen - xs.size)(0)
        val c = positions ++ Array.fill[Double](maxSeqLen - xs.size)(pad)
        val d = masks     ++ Array.fill[Double](maxSeqLen - xs.size)(0) // fix a bug: 0 instead of 1 for padded tokens
        Vectors.dense(a ++ b ++ c ++ d)
      }
    }

    f(_)
  }

  override protected def outputDataType: DataType = VectorType
}

object Sequencer4BERT extends DefaultParamsReadable[Sequencer4BERT] {
  override def load(path: String): Sequencer4BERT = super.load(path)
}