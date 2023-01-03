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
  * using a dictionary. The unknown tokens are indexed by -1.
  *
  * phuonglh@gmail.com
  */
class Sequencer(val uid: String, val dictionary: Map[String, Int]) extends UnaryTransformer[Seq[String], Vector, Sequencer]
  with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Int]]] = None

  def this(dictionary: Map[String, Int]) = {
    this(Identifiable.randomUID("seq"), dictionary)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
  }

  override protected def createTransformFunc: Seq[String] => Vector = {
    def f(xs: Seq[String]): Vector = {
      val a = xs.map(x => dictionaryBr.get.value.getOrElse(x, -1).toDouble).toArray
      Vectors.dense(a)
    }

    f(_)
  }

  override protected def outputDataType: DataType = VectorType
}

object Sequencer extends DefaultParamsReadable[Sequencer] {
  override def load(path: String): Sequencer = super.load(path)
}