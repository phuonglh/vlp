package vlp.woz.jsl

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.DataType


/**
  * A sequence vectorizer transforms a sequence of labels into a sequence of doubles
  * using a dictionary. No padding is applied.
  *
  * phuonglh@gmail.com
  */
class Sequencer(val uid: String, val dictionary: Map[String, Double]) 
  extends UnaryTransformer[Seq[String], Vector, Sequencer] with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Double]]] = None

  def this(dictionary: Map[String, Double]) = {
    this(Identifiable.randomUID("labelSeq"), dictionary)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
  }

  override protected def createTransformFunc: Seq[String] => Vector = {
    def f(xs: Seq[String]): Vector = {
      val a = xs.map(x => dictionaryBr.get.value.getOrElse(x, 0d).toDouble).toArray
      Vectors.dense(a)
    }

    f(_)
  }

  override protected def outputDataType: DataType = VectorType
}

object Sequencer extends DefaultParamsReadable[Sequencer] {
  override def load(path: String): Sequencer = super.load(path)
}