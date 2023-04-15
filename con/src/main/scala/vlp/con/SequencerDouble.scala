package vlp.con

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataType, ArrayType, StringType}


/**
  * A sequence vectorizer transforms a sequence of double indices into a sequence of string labels
  * using a dictionary. No padding is applied. This is the inverse transformer of [[SequencerNER]].
  *
  * phuonglh@gmail.com
  */
class SequencerDouble(val uid: String, val dictionary: Map[Double, String])
  extends UnaryTransformer[Seq[Double], Seq[String], SequencerDouble] with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[Double, String]]] = None

  def this(dictionary: Map[Double, String]) = {
    this(Identifiable.randomUID("doubleSeq"), dictionary)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
  }

  override protected def createTransformFunc: Seq[Double] => Seq[String] = {
    def f(xs: Seq[Double]): Seq[String] = {
      xs.map(x => dictionaryBr.get.value.getOrElse(x, "NA"))
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(StringType, false)
}

object SequencerDouble extends DefaultParamsReadable[SequencerDouble] {
  override def load(path: String): SequencerDouble = super.load(path)
}