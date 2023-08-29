package vlp.vdg

/**
  * @author Phuong LE-HONG
  *
  * Detector of different word shapes.
  *
  */
object WordShape {
  val docNo = """\b(\d+[\p{Lu}-]*/)+[\p{Lu}-]+""".r
  val lowercase = """[\p{javaLowerCase}_]+""".r
  val capitalized = """\p{javaUpperCase}[\p{javaLowerCase}_]*""".r
  val capitalizedWithPeriod = """(\p{javaUpperCase}\p{javaLowerCase}?\.)+\p{javaUpperCase}?""".r
  val allcaps = """\b\p{Lu}+([\s_]\p{Lu}+)*\b""".r
  val mixedCase = """\p{javaLowerCase}+\p{javaUpperCase}\p{javaLowerCase}+""".r
  val endsInDigit = """[\p{javaUpperCase}\p{javaLowerCase}]+\d+""".r
  val containsHyphen = """[\p{javaUpperCase}\p{javaLowerCase}]+\-[\p{javaUpperCase}\p{javaLowerCase}]+""".r
  val number = """[\+\-]?([0-9]*)?[0-9]+([\.,]\d+)*""".r
  val percentage = """[\+\-]?([0-9]*)?[0-9]+([\.,]\d+)*%""".r
  val date = """(\d\d\d\d)[-/\.](\d?\d)[-/\.](\d?\d)|((\d?\d)[-/\.])?(\d?\d)[-/\.](\d\d\d\d)""".r
  val name = """(\p{javaUpperCase}\p{javaLowerCase}+)([-_&,\s]+(\p{javaUpperCase}\p{javaLowerCase}+))+""".r
  val code = """\d+\p{javaUpperCase}""".r
  val weight = """[\+\-]?([0-9]*)?[0-9]+([\.,]\d+)*(Kg|kg|lbs|kilograms|kilogram|kilos|kilo|pounds|pound)""".r
  val punctuation = """[.,?!;:\"…/”“″=^▪•<>&«\])(\[\u0022\u200b\ufeff+-]+""".r
  val email = """(\w[-._%:\w]*@\w[-._\w]*\w\.\w{2,3})""".r
  val url = """(((\w+)\://)+[a-zA-z][\-\w]*\w+(\.\w[\-\w]*)+(/[\w\-~]+)*(\.\w+)?(/?)(\?(\w+=[\w%~]+))*(&(\w+=[\w%~]+))*|[a-z]+((\.)\w+)+)""".r
  val date1 = """\b(([12][0-9]|3[01]|0*[1-9])[-/.](1[012]|0*[1-9])[-/.](\d{4}|\d{2})|(1[012]|0*[1-9])[-/.]([12][0-9]|3[01]|0*[1-9])[-/.](\d{4}|\d{2}))\b""".r
  val date2 = """\b(1[012]|0*[1-9])[-/](\d{4}|\d{2})\b""".r
  val date3 = """\b([12][0-9]|3[01]|0*[1-9])[-/](1[012]|0*[1-9])\b""".r
  val date4 = """\b([Nn]gày)([\s_]+)\d+([\s_]+)tháng([\s_]+)\d+([\s_]+)năm([\s_]+)(\d+)\b""".r
  val time = """\b\d{1,2}:\d{1,2}\b""".r
  val numberSeq = """\+?\d+(([\s._-]+\d+)){2,}\b""".r


  /**
    * Detects the shape of a word and returns its shape string or empty.
    * @param word
    * @return word shape
    */
  def shape(word: String): String = {
    word match {
      case docNo(_*) => "docNo"
      case url(_*) => "url"
      case email(_*) => "email"
      case lowercase(_*) => "lower"
      case capitalized(_*) => "capitalized"
      case capitalizedWithPeriod(_*) => "capitalizedWithPeriod"
      case allcaps(_*) => "allcaps"
      case mixedCase(_*) => "mixedCase"
      case endsInDigit(_*) => "endsInDigit"
      case containsHyphen(_*) => "containsHyphen"
      case date1(_*) => "date" // should goes after number (but we want to recognize 12.2004 as a date.)
      case date2(_*) => "date"
      case date3(_*) => "date"
      case date4(_*) => "date"
      case time(_*) => "time"
      case numberSeq(_*) => "numSeq"
      case number(_*) => "number"
      case percentage(_*) => "percentage"
      case code(_*) => "code"
      case name(_*) => "name"
      case weight(_*) => "weight"
      case punctuation(_*) => "punctuation"
      case _ => ""
    }
  }

  def isNamedEntity(word: String): Boolean = {
    shape(word) match {
      case "capitalized" => true
      case "allcaps" => true
      case "name" => true
      case _ => false
    }
  }

  def normalize(word: String): String = {
    val s = shape(word)
    s match {
      case "lower" => word
      case "" => word
      case "allcaps" => word
      case "capitalized" => word
      case "capitalizedWithPeriod" => word
      case "mixedCase" => word
      case "endsInDigit" => word
      case "containsHyphen" => word
      case "name" => word
      case _ => '<' + s + '>'
    }
  }


  def main(args: Array[String]): Unit = {
    val words = List("love", "Washington", "ABC", "H.", "A9", "F35", "H-P", "UBKT", "TƯ",
      "ĐHTH", "eBay", "iPhone", "4,120", "-4.120", "1996-08-22", "20-10-1980", "U.S.",
      "Hà_Nội", "Huế", "Đà_Nẵng", "Buôn_Mê_Thuột", "92.9kg", "93.3", "12.24.2000", "30",
      "Modesto, California", "U.S.", "U.S.A", "U.S.A.", "phuonglh@gmail.com", "http://vlp.group/lhp", "http://phuonglh.com/", 
      "vlp.group", "vlp.group/vlp", "4158/QĐ-UBND", "ngày_05_tháng_2_năm_2015", "26/2016/TT-BXD", "27/NQ-HDND", "95%", "10.3%",
      "119/2009/QD-", "119/2009/QĐ-", "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "CỘNG_HÒA_XÃ_HỘI_CHỦ_NGHĨA_VIỆT_NAM"
    )
    words.foreach(e => {
      println(e + " => shape = " + WordShape.shape(e) + ", normalize = " + normalize(e))
    })
  }
}