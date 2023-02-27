package vlp.woz

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

/**
  * phuonglh@gmail.com
  *
  * Transform a sequence of words to a sequence of shapes. If a word does not have 
  * interested shapes, then it is left unchanged: "abc" => "abc", "12" => "[NUMBER]", etc.
  * 
  */
class WordShaper(override val uid: String) extends UnaryTransformer[Seq[String], Seq[String], WordShaper] with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("shaper"))

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    _.map(x => {
      val sh = WordShape.shape(x)
      if (sh.isEmpty()) x else s"[${sh.toUpperCase}]"
    })
  }

  override protected def outputDataType: DataType = ArrayType(StringType, false)
}

object WordShaper extends DefaultParamsReadable[WordShaper] {
  override def load(path: String): WordShaper = super.load(path)
}

/**
  * 
  * Detector of different word shapes.
  *
  */
object WordShape {
  val allcaps = """\b\p{Lu}+([\s_]\p{Lu}+)*\b""".r
  val number = """[\+\-]?([0-9]*)?[0-9]+([\.,]\d+)*""".r
  val percentage = """[\+\-]?([0-9]*)?[0-9]+([\.,]\d+)*%""".r
  val punctuation = """[.,?!;:\"…/”“″'=^▪•<>&«\])(\[\u0022\u200b\ufeff+-]+""".r
  val email = """(\w[-._%:\w]*@\w[-._\w]*\w\.\w{2,3})""".r
  val url = """(((\w+)\://)+[a-zA-z][\-\w]*\w+(\.\w[\-\w]*)+(/[\w\-~]+)*(\.\w+)?(/?)(\?(\w+=[\w%~]+))*(&(\w+=[\w%~]+))*|[a-z]+((\.)\w+)+)""".r
  val date = """(\d\d\d\d)[-/\.](\d?\d)[-/\.](\d?\d)|((\d?\d)[-/\.])?(\d?\d)[-/\.](\d\d\d\d)""".r
  val date1 = """\b(([12][0-9]|3[01]|0*[1-9])[-/.](1[012]|0*[1-9])[-/.](\d{4}|\d{2})|(1[012]|0*[1-9])[-/.]([12][0-9]|3[01]|0*[1-9])[-/.](\d{4}|\d{2}))\b""".r
  val date2 = """\b(1[012]|0*[1-9])[-/](\d{4}|\d{2})\b""".r
  val date3 = """\b([12][0-9]|3[01]|0*[1-9])[-/](1[012]|0*[1-9])\b""".r
  val time = """\b\d{1,2}:\d{1,2}\b""".r
  val numberSeq = """\+?\d+(([\s._-]+\d+)){2,}\b""".r


  /**
    * Detects the shape of a word and returns its shape string or empty.
    * @param word
    * @return word shape
    */
  def shape(word: String): String = {
    word match {
      case email(_*) => "email"
      case url(_*) => "url"
      case allcaps(_*) => "allcaps"
      case date1(_*) => "date" // should goes after number (but we want to recognize 12.2004 as a date.)
      case date2(_*) => "date"
      case date3(_*) => "date"
      case date(_*) => "date"
      case time(_*) => "time"
      case numberSeq(_*) => "numSeq"
      case number(_*) => "number"
      case percentage(_*) => "percentage"
      case punctuation(_*) => "punctuation"
      case _ => ""
    }
  }
}