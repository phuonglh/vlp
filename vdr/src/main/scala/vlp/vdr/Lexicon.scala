package vlp.vdr

/**
  * phuonglh, 1/8/18, 3:04 PM
  */
object Lexicon {
  final val words: Set[String] = IO.readSentences("/lexicon.txt", true).toSet
  final val syllables: Set[String] = words.flatMap(word => word.split("\\s+")).map(_.toLowerCase)
  final val punctuations = Map(
    "." -> "\\.", 
    "!" -> "\\!", 
    "," -> ",", 
    "?" -> "\\?", 
    ";" -> ";", 
    ":" -> ":", 
    "(" -> "\\(", 
    ")" -> "\\)", 
    "'" -> "'", 
    "\"" -> "\"",
    " " -> " "
  )
  
}
