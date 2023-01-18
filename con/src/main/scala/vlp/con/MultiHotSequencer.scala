

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.DataType


/**
  * A multi-hot sequence vectorizer transforms a sequence of tokens into a sequence of character indices
  * using a dictionary. This transformer pads or 
  * truncate long sentence to a given `maxSequenceLength`. See the character model for detail.
  * Given a syllable "khanh", this transformer converts it to a multi-hot vector of length 3*|V|, 
  * where |V| is the size of a dictionary (alphabet), b::i::e, where b and e are two one-hot vectors, 
  * and e is a multi-hot vector representing a bag-of-character "{h, a, n}".  
  * 
  * phuonglh@gmail.com
  */
class MultiHotSequencer(val uid: String, val dictionary: Map[String, Int], maxSequenceLength: Int, padding: Float) 
  extends UnaryTransformer[Seq[String], Vector, MultiHotSequencer] with DefaultParamsWritable {

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
    def g(x: String): Array[Double] = {
      val b = x.charAt(0).toString()
      val e = if (x.size >= 2) x.charAt(x.size-1).toString() else "NA"
      val m = if (x.size >= 3) {
        x.substring(1, x.size-1)
      } else "NA"
      val vocab = dictionaryBr.get.value
      val vb = Array.fill[Double](vocab.size)(0)
      vb(vocab(b)) = 1.0
      val ve = Array.fill[Double](vocab.size)(0)
      ve(vocab(e)) = 1.0
      val vm = Array.fill[Double](vocab.size)(0)
      m.foreach(c => vm(c) = 1.0)
      return vb ++ vm ++ ve
    }

    def f(xs: Seq[String]): Vector = {      
      // truncate or pad
      if (xs.size >= maxSeqLen) {
        val ys = xs.take(maxSeqLen)
        val vs = ys.map(x => g(x))
        val v = vs.reduce((a, b) => Array.concat(a, b))
        Vectors.dense(v)
      } else {
        val vs = xs.map(x => g(x))
        val v = vs.reduce((a, b) => Array.concat(a, b))
        val vocab = dictionaryBr.get.value
        val b = v ++ Array.fill[Double]((maxSeqLen - xs.size)*3*vocab.size)(pad)
        Vectors.dense(b)
      }
    }

    f(_)
  }

  override protected def outputDataType: DataType = VectorType
}

object MultiHotSequencer extends DefaultParamsReadable[MultiHotSequencer] {
  override def load(path: String): MultiHotSequencer = super.load(path)
}