package vlp.tok

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.{BooleanParam, Param, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StringType
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.SparkSession

/**
  * phuonglh
  *
  * June 2019
  */
trait TokenizerTransformerParams extends Params {
  val convertPunctuation: Param[Boolean] = new BooleanParam(this, "convert punctuations to PUNCT", "convert punctuations")
  val convertNumber: Param[Boolean] = new BooleanParam(this, "convert number to [NUM]", "convert numbers")
  val toLowercase: Param[Boolean] = new BooleanParam(this, "convert text to lowercase", "make text lowercase")
  val splitSentences: Param[Boolean] = new BooleanParam(this, "split text into sentences", "split text into sentences")

  def getConvertPunctuation: Boolean = $(convertPunctuation)
  def setConvertPunctuation(value: Boolean): this.type  = set(convertPunctuation, value)
  def getConvertNumber: Boolean = $(convertNumber)
  def setConvertNumber(value: Boolean): this.type  = set(convertNumber, value)
  def getToLowercase: Boolean = $(toLowercase)
  def setToLowercase(value: Boolean): this.type  = set(toLowercase, value)
  def getSplitSentences: Boolean = $(splitSentences)
  def setSplitSentences(value: Boolean): this.type  = set(splitSentences, value)

  setDefault(convertPunctuation -> false, convertNumber -> false, toLowercase -> false, splitSentences -> false)
}

class TokenizerTransformer(override val uid: String) extends UnaryTransformer[String, String, TokenizerTransformer]
  with TokenizerTransformerParams with DefaultParamsWritable {
  
  val tokenizerBr: Option[Broadcast[Tokenizer]] = Some(SparkSession.getActiveSession.get.sparkContext.broadcast(new Tokenizer()))

  def this() = this(Identifiable.randomUID("viTok"))

  override protected def createTransformFunc: (String) => String = { text =>

    def f(text: String): String = {
      val tokens = tokenizerBr.get.value.tokenize(text)
      val output = tokens.map(_._3)
      // convert punct
      val us = if ($(convertPunctuation))
        output.map { token => TokenizerTransformer.convertPunct(token) }
      else output
      // convert number
      val vs = if ($(convertNumber)) 
        us.map { token => TokenizerTransformer.convertNum(token) }
      else us
      if ($(toLowercase))
        vs.mkString(" ").toLowerCase
      else vs.mkString(" ")
    }

    if (!getSplitSentences) {
      f(text)
    } else {
      val sb = new StringBuilder(1024)
      val sentences = SentenceDetection.run(text)
      sentences.foreach(text => {
        sb.append(f(text))
        sb.append(" ")
      })
      sb.toString.trim
    }
  }

  override protected def outputDataType = StringType
}

object TokenizerTransformer extends DefaultParamsReadable[TokenizerTransformer] {
  final val punctuations = Array(",", ".", ":", ";", "?", "!", "\"", "'", "/", "...", "-", "LBKT", "RBKT", "--", "``", "''", ")", "(")

  def convertPunct(token: String): String = if (punctuations.contains(token)) "PUNCT" else token.replaceAll(",", ".")

  def convertNum(token: String): String = token match {
    case WordShape.number(_*) => "[NUM]"
    case WordShape.percentage(_*) => "[NUM]"
    case _ => token
  }

  override def load(path: String): TokenizerTransformer = super.load(path)
}
