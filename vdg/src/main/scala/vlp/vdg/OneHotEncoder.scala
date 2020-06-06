package vlp.vdg

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{ArrayType, DataType, FloatType}

/**
  * phuonglh, 6/18/18, 12:09 PM
  * 
  * A transformer which transforms sequence of words to sequence of one hot vector using a
  * vocabulary.
  */
class OneHotEncoder(val uid: String, val vocabulary: Array[String]) 
  extends UnaryTransformer[Seq[String], Array[Array[Float]], OneHotEncoder] with DefaultParamsWritable with EncoderParams {
  
  var vocabularyBr: Option[Broadcast[Array[String]]] = None

  def this(uid: String) = this(uid, Array.empty[String])
  
  def this(vocabulary: Array[String]) = {
    this(Identifiable.randomUID("oneHotEncoder"), vocabulary)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    vocabularyBr = Some(sparkContext.broadcast(vocabulary))
  }

  override protected def createTransformFunc: Seq[String] => Array[Array[Float]] = {
    def f(xs: Seq[String]): Seq[Array[Float]] = {
      val n = $(sequenceLength)
      val paddedTokens = if (xs.length > n) {
        // truncate this long input, either by drop the beginning or the end
        if ($(truncated) == "pre")
          xs.slice(xs.length - n, xs.length)
        else xs.slice(0, n)
      } else {
        // pad the end of this short input
        xs ++ Array.fill[String](n - xs.length)("$NA$")
      }
      // transform paddedTokens to one-hot vectors
      paddedTokens.map { x =>
        val v = Array.fill[Float]($(numFeatures))(0)
        if (vocabularyBr.get.value.contains(x)) {
          val j = vocabularyBr.get.value.indexOf(x)
          v(j) = 1f
        }
        v
      }
    }
    
    f(_).toArray
  }

  override protected def outputDataType: DataType = ArrayType(ArrayType(FloatType, false), false)
  
}

object OneHotEncoder extends DefaultParamsReadable[OneHotEncoder] {
  override def load(path: String): OneHotEncoder = super.load(path)
}