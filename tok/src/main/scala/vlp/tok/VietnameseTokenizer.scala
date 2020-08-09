package vlp.tok

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.{BooleanParam, Param, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.ArrayType

/**
  * phuonglh
  * A Vietnamese tokenizer which acts the same Spark tokenizer.
  * August 2020
  */
class VietnameseTokenizer(override val uid: String) extends UnaryTransformer[String, Seq[String], VietnameseTokenizer]
  with vlp.tok.TokenizerTransformerParams with DefaultParamsWritable {
  
  def this() = this(Identifiable.randomUID("vlp.tok"))

  override protected def createTransformFunc: (String) => Seq[String] = { text =>

    def f(text: String): Seq[String] = {
      val tokens = vlp.tok.Tokenizer.tokenize(text)
      val output = tokens.map(_._3)
      // convert punct
      val us = if ($(convertPunctuation))
        output.map { token => VietnameseTokenizer.convertPunct(token) }
      else output
      // convert number
      val vs = if ($(convertNumber)) 
        us.map { token => VietnameseTokenizer.convertNum(token) }
      else us
      if ($(toLowercase))
        vs.map(_.toLowerCase())
      else vs
    }

    if (!getSplitSentences) {
      f(text)
    } else {
      val sentences = vlp.tok.SentenceDetection.run(text)
      sentences.flatMap(text => f(text))
    }
  }

  override protected def outputDataType = ArrayType(StringType, false)
}

object VietnameseTokenizer extends DefaultParamsReadable[VietnameseTokenizer] {
  final val punctuations = Array(",", ".", ":", ";", "?", "!", "\"", "'", "/", "...", "-", "LBKT", "RBKT", "--", "``", "''", ")", "(")

  def convertPunct(token: String): String = if (punctuations.contains(token)) "PUNCT" else token.replaceAll(",", ".")

  def convertNum(token: String): String = token match {
    case vlp.tok.WordShape.number(_*) => "[NUM]"
    case vlp.tok.WordShape.percentage(_*) => "[NUM]"
    case _ => token
  }

  override def load(path: String): VietnameseTokenizer = super.load(path)
}
