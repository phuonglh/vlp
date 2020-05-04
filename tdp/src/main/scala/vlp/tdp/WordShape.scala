package vlp.tdp

/**
 * @author Phuong LE-HONG
 * <p>
 * Sep 17, 2016, 11:13:37 AM
 * <p>
 * Detector of different word shapes.
 * 
 */
object WordShape {
  val lowercase = """[\p{javaLowerCase}_]+""".r
  val capitalized = """\p{javaUpperCase}[\p{javaLowerCase}_]*""".r
  val capitalizedWithPeriod = """(\p{javaUpperCase}\p{javaLowerCase}?\.)+""".r 
  val allcaps = """\p{javaUpperCase}+""".r
  val mixedCase = """\p{javaLowerCase}+\p{javaUpperCase}\p{javaLowerCase}+""".r
  val endsInDigit = """[\p{javaUpperCase}\p{javaLowerCase}]+\d+""".r 
  val containsHyphen = """[\p{javaUpperCase}\p{javaLowerCase}]+\-[\p{javaUpperCase}\p{javaLowerCase}]+""".r
  val number = """[\+\-]?([0-9]*)?[0-9]+([\.,]\d+)*""".r
  val date = """(\d\d\d\d)[-/\.](\d?\d)[-/\.](\d?\d)|((\d?\d)[-/\.])?(\d?\d)[-/\.](\d\d\d\d)""".r
  val name = """(\p{javaUpperCase}[\p{javaLowerCase}_]+)+""".r
  val code = """\d+\p{javaUpperCase}""".r
  val weight = """[\+\-]?([0-9]*)?[0-9]+([\.,]\d+)*(Kg|kg|lbs|kilograms|kilogram|kilos|kilo|pounds|pound)""".r
  val punctuations = """[,.:;!?"']+""".r
  /**
   * Detects the shape of a word and returns its shape string or empty.
   * @param word
   * @return word shape
   */
  def shape(word: String): String = {
    word match {
      case lowercase(_*) => "lower"
      case capitalized(_*) => "capitalized"
      case capitalizedWithPeriod(_*) => "capitalizedWithPeriod"
      case allcaps(_*) => "allcaps"
      case mixedCase(_*) => "mixedCase"  
      case endsInDigit(_*) => "endsInDigit"
      case containsHyphen(_*) => "containsHyphen"
      case date(_*) => "date" // should goes after number (but we want to recognize 12.2004 as a date.)
      case number(_*) => "number"
      case code(_*) => "code"
      case name(_*) => "name"
      case weight(_*) => "weight"
//      case punctuations(_*) => "punctuation"
      case _ => ""
    }
  }
  
  def main(args: Array[String]): Unit = {
    val words = List("love", "Washington", "ABC", "H.", "A9", "F35", "H-P", "UBKT", "TƯ", 
        "ĐHTH", "eBay", "iPhone", "4,120", "-4.120", "1996-08-22", "20-10-1980", "U.S.", 
        "Hà_Nội", "Huế", "Đà_Nẵng", "Buôn_Mê_Thuột", "92.9kg", "93.3", "12.24.2000", "30", ".", "??")
    words.map(WordShape.shape(_)).foreach(println)
  }
}
