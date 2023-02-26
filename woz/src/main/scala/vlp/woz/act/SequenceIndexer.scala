package vlp.woz.act

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataType, ArrayType, DoubleType}


/**
  * A sequence vectorizer transforms a sequence of tokens into a sequence of indices
  * using a dictionary. The output sequence has the same length as the input sequence.
  *
  * phuonglh@gmail.com
  */
class SequenceIndexer(val uid: String, val dictionary: Map[String, Int]) 
  extends UnaryTransformer[Seq[String], Seq[Double], SequenceIndexer] with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Int]]] = None

  def this(dictionary: Map[String, Int]) = {
    this(Identifiable.randomUID("seqInd"), dictionary)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
  }

  override protected def createTransformFunc: Seq[String] => Seq[Double] = {
    _.map(x => dictionaryBr.get.value.getOrElse(x, 0).toDouble)
  }

  override protected def outputDataType: DataType = ArrayType(DoubleType, false)
}

object SequenceIndexer extends DefaultParamsReadable[SequenceIndexer] {
  override def load(path: String): SequenceIndexer = super.load(path)
}