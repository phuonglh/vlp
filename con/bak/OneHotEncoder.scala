package vlp.con

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{ArrayType, DataType, FloatType}
import org.apache.spark.ml.param.{IntParam, Param, ParamValidators, Params}

/**
  * Some parameters of a sequence encoder.
  *
  * See [[OneHotEncoder]]
  */
trait OneHotEncoderParams extends Params {
  val numFeatures: IntParam = new IntParam(this, "number of features", "domain dimension", ParamValidators.gt(0))
  val sequenceLength: IntParam = new IntParam(this, "sequenceLength", "max number of tokens of a text", ParamValidators.gt(2))
  val truncated: Param[String] = new Param[String](this, "truncated", "how to truncate sequence", ParamValidators.inArray(Array("pre", "post")))
  
  def getNumFeatures: Int = $(numFeatures)
  def setNumFeatures(value: Int): this.type = set(numFeatures, value)
  def getSequenceLength: Int = $(sequenceLength)
  def setSequenceLength(value: Int): this.type = set(sequenceLength, value)
  def getTruncated: String = $(truncated)
  def setTruncated(value: String): this.type = set(truncated, value)
  
  setDefault(numFeatures -> 2048, sequenceLength -> 128, truncated -> "post")
}

/**
  * phuonglh@gmail.com
  * 
  * A transformer which transforms sequence of words to sequence of one hot vectors using a
  * dictionary. If there are `n` words in a sequence and the dictionary has `m` entries, then 
    this produces a vector of `n times m` elements, in order.
  */
class OneHotEncoder(val uid: String, val dict: Map[String,Int]) 
  extends UnaryTransformer[Seq[String], Array[Float], OneHotEncoder] with DefaultParamsWritable with OneHotEncoderParams {
  
  var dictBr: Option[Broadcast[Map[String,Int]]] = None

  def this(uid: String) = this(uid, Map.empty[String, Int])
  
  def this(dict: Map[String,Int]) = {
    this(Identifiable.randomUID("ohe"), dict)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictBr = Some(sparkContext.broadcast(dict))
  }

  override protected def createTransformFunc: Seq[String] => Array[Float] = {
    def f(xs: Seq[String]): Array[Float] = {
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
      val vs = paddedTokens.map { x =>
        val v = Array.fill[Float]($(numFeatures))(0)
        val j = dictBr.get.value.getOrElse(x, 0)
        v(j) = 1f
        v
      }
      vs.toArray.flatMap(_.toList)
    }
    
    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(FloatType, false)
  
}

object OneHotEncoder extends DefaultParamsReadable[OneHotEncoder] {
  override def load(path: String): OneHotEncoder = super.load(path)
}