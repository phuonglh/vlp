package vlp.vdg

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{ArrayType, DataType, IntegerType}


/**
  * A sequence vectorizer transforms a sequence of tokens into a sequence of indices
  * using a dictionary. The unknown tokens are indexed by 0. The first token is index
  * by 1 and the last token is index by size(dictionary).
  *
  * phuonglh@gmail.com
  */
class SequenceVectorizer(val uid: String, val dictionary: Map[String, Int]) extends UnaryTransformer[Seq[String], Array[Int], SequenceVectorizer]
  with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Int]]] = None

  def this(dictionary: Map[String, Int]) = {
    this(Identifiable.randomUID("seqVec"), dictionary)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
  }

  override protected def createTransformFunc: Seq[String] => Array[Int] = {
    def f(xs: Seq[String]): Array[Int] = {
      xs.map(x => dictionaryBr.get.value.getOrElse(x, -1) + 1).toArray
    }

    f(_)
  }

  override protected def outputDataType: DataType = new ArrayType(IntegerType, false)
}

object SequenceVectorizer extends DefaultParamsReadable[SequenceVectorizer] {
  override def load(path: String): SequenceVectorizer = super.load(path)
}