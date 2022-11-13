package vlp.vdg

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{ArrayType, DataType, IntegerType}

/**
  * phuonglh, 10/15/18
  * 
  * A transformer which transforms sequence of words to sequence of integers using their indices in
  * a vocabulary.
  */
class TokenEncoder(val uid: String, val vocabulary: Array[String])
  extends UnaryTransformer[Seq[String], Array[Int], TokenEncoder] with DefaultParamsWritable with EncoderParams {
    
  var vocabularyBr: Option[Broadcast[Array[String]]] = None

  def this(uid: String) = this(uid, Array.empty[String])
  
  def this(vocabulary: Array[String]) = {
    this(Identifiable.randomUID("tokenEncoder"), vocabulary)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    vocabularyBr = Some(sparkContext.broadcast(vocabulary))
  }
  
  override protected def createTransformFunc: Seq[String] => Array[Int] = {
    def f(xs: Seq[String]): Seq[Int] = {
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

      // convert each token into its index or -1 if this is an unknown token
      paddedTokens.map { x =>
        if (vocabularyBr.get.value.contains(x)) {
          vocabularyBr.get.value.indexOf(x)
        } else -1
      }
    }
    
    f(_).toArray
  }
  
  override protected def outputDataType: DataType = ArrayType(IntegerType, false)
}

object TokenEncoder extends DefaultParamsReadable[TokenEncoder] {
  override def load(path: String): TokenEncoder = super.load(path)
}