package vlp.vdg

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{ArrayType, DataType, IntegerType}

/**
  * Transforms a sequence of tokens into a sequence of sequences of characters.
  * For example, ["Joe", "loves", "Mie"] is transformed to
  * [["J", "o", "e"], ["l", "o", "v", "e", "s"], ["M", "i", "e"]].
  * To facilitate the training, we further convert each character to an integer index
  * using a character dictionary. Note that unknown characters are converted to 0.
  */

class CharSequencer(val uid: String, charDictionary: Map[String, Int]) extends UnaryTransformer[Seq[String], Seq[Seq[Int]], CharSequencer]
  with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Int]]] = None

  def this(charDictionary: Map[String, Int]) = {
    this(Identifiable.randomUID("charSeq"), charDictionary)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(charDictionary))
  }

  override protected def createTransformFunc: Seq[String] => Seq[Seq[Int]] = {
    def f(xs: Seq[String]): Seq[Seq[Int]] = {
      xs.map(x => x.map(c => dictionaryBr.get.value.getOrElse(c.toString, -1) + 1))
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(ArrayType(IntegerType, false), false)
}

object CharSequencer extends DefaultParamsReadable[CharSequencer] {
  override def load(path: String): CharSequencer = super.load(path)
}