package vlp.tpm;

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, StringType}

/**
  * Converts term indices into terms.
  *
  * phuonglh, phuonglh@gmail.com
  * @param uid
  * @param vocabulary
  */
class TermIndexToString(override val uid: String, val vocabulary: Array[String]) extends UnaryTransformer[Seq[Int], Seq[String], TermIndexToString] with DefaultParamsWritable {
  var map = Map[Int, String]()

  def this(vocabulary: Array[String]) = {
    this(Identifiable.randomUID("idxToStr"), vocabulary)
    map = (0 until vocabulary.size).zip(vocabulary).toMap
  }

  override protected def createTransformFunc: (Seq[Int] => Seq[String]) = {
    def f(indices: Seq[Int]): Seq[String] = {
      indices.map(map.getOrElse(_, ""))
    }

    f(_)
  }

  override protected def outputDataType = new ArrayType(StringType, false)
}

object TermIndexToString extends DefaultParamsReadable[TermIndexToString] {
  override def load(path: String): TermIndexToString = super.load(path)
}
