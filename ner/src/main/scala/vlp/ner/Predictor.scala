package vlp.ner

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataType
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.types.StringType

/**
  * A transformer which transforms prediction result of a [[NNModel]] to a label sequence. The prediction column of a DLModel 
  * is a sequence of maxSequenceLength x labelSize real-valued numbers. We need a label dictionary to map indices to label strings.
  * 
  * phuonglh@gmail.com
  */
class Predictor(val uid: String, val labelMap: Map[Int, String], maxSequenceLength: Int) extends UnaryTransformer[Seq[Double], Seq[String], Predictor]
  with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[Int, String]]] = None

  def this(labelMap: Map[Int, String], maxSequenceLength: Int) = {
    this(Identifiable.randomUID("predictor"), labelMap, maxSequenceLength)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(labelMap))
  }

  override protected def createTransformFunc: Seq[Double] => Seq[String] = {
    val labels = dictionaryBr.get.value
    def f(xs: Seq[Double]): Seq[String] = {
      val slices = xs.sliding(labelMap.size, labelMap.size).toList
      val maxIndices = slices.map(distribution => distribution.zipWithIndex.sortBy(_._1).last._2)
      maxIndices.map(i => labels(i))
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(StringType, false)
}

object Predictor extends DefaultParamsReadable[Predictor] {
  override def load(path: String): Predictor = super.load(path)
}