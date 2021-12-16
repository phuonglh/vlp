package vlp.ner

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataType
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

/**
  * A sequence vectorizer transforms a sequence of tokens into a sequence of indices
  * using a dictionary. The unknown tokens are indexed by 0. The first token is index
  * by 1 and the last token is index by size(dictionary).
  * 
  * See also [[vlp.vdg.SequenceVectorizer]]
  * See also [[vlp.nli.SequenceVectorizer]]
  *
  * phuonglh@gmail.com
  */
class SequenceVectorizer(val uid: String, val dictionary: Map[String, Int], maxSequenceLength: Int, binary: Boolean, paddingValue: Int) 
    extends UnaryTransformer[Seq[String], Vector, SequenceVectorizer]
        with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Int]]] = None

  def this(dictionary: Map[String, Int], maxSequenceLength: Int, binary: Boolean = false, paddingValue: Int = 1) = {
    this(Identifiable.randomUID("seqVec"), dictionary, maxSequenceLength, binary, paddingValue)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
  }

  override protected def createTransformFunc: Seq[String] => Vector = {
    val dict = dictionaryBr.get.value
    def f(xs: Seq[String]): Vector = {
      if (!binary) {
        val indices = xs.map(x => dict.getOrElse(x, 0) + 1.0).toArray
        val values = if (indices.size >= maxSequenceLength) 
          indices.take(maxSequenceLength) 
        else indices ++ Array.fill(maxSequenceLength - indices.size)(paddingValue.toDouble)
        Vectors.dense(values)
      } else {
        val values = xs.map(x => if (dict.contains(x)) 1.0 else 0.0).toArray
        if (values.size >= maxSequenceLength)
          Vectors.dense(values.take(maxSequenceLength))
        else Vectors.dense(values ++ Array.fill(maxSequenceLength - values.size)(0.0))
      }
    }

    f(_)
  }

  override protected def outputDataType: DataType = VectorType
}

object SequenceVectorizer extends DefaultParamsReadable[SequenceVectorizer] {
  override def load(path: String): SequenceVectorizer = super.load(path)
}