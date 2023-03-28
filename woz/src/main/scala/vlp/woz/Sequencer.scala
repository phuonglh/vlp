package vlp.woz

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataType, ArrayType, DoubleType}


/**
  * A sequence vectorizer transforms a sequence of labels into a sequence of indices
  * using a dictionary. No padding is applied.
  *
  * phuonglh@gmail.com
  */
class Sequencer(val uid: String, val dictionary: Map[String, Int]) 
  extends UnaryTransformer[Seq[String], Seq[Double], Sequencer] with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Int]]] = None

  def this(dictionary: Map[String, Int]) = {
    this(Identifiable.randomUID("labelSeq"), dictionary)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
  }

  override protected def createTransformFunc: Seq[String] => Seq[Double] = {
    def f(xs: Seq[String]): Seq[Double] = {
      xs.map(x => dictionaryBr.get.value.getOrElse(x, 0).toDouble).toArray
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(DoubleType, false)
}

object Sequencer extends DefaultParamsReadable[Sequencer] {
  override def load(path: String): Sequencer = super.load(path)
}