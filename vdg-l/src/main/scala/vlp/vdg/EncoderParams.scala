package vlp.vdg

import org.apache.spark.ml.param.{IntParam, Param, ParamValidators, Params}

/**
  * Some parameters of a sequence encoder.
  *
  * See [[OneHotEncoder]], [[TokenEncoder]]
  */
trait EncoderParams extends Params {
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
