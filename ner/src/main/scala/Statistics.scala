import vlp.ner.CorpusReader
import breeze.linalg.max

/**
  * Simple utilty to get the summary statistics of the NER corpus.
  * 
  */
object Statistics {
  def main(args: Array[String]): Unit = {
    val sentences = CorpusReader.readCoNLL("dat/ner/vie/vie.train")
    val lengths = sentences.map(_.tokens.size)
    val counts = lengths.groupBy(identity).mapValues(_.size)
    val histogram = counts.toArray.sortBy(_._1).toList
    val st = histogram.map(p => p._1 + "\t" + p._2)
    println(st.mkString("\n"))
    val maxLength = 60
    val total = histogram.map(_._2).sum
    val sumToMaxLength = histogram.filter(_._1 <= maxLength).map(_._2).sum
    val percentage = sumToMaxLength.toFloat / total * 100.0f
    println(percentage)
  }
}
