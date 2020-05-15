package vlp.vdr

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, StringType}

import scala.collection.mutable.ListBuffer

/**
  * phuonglh, 10/31/17, 18:52
  * 
  * Converts a string into a sequence of characters.
  */
class Sequencer(override val uid: String) extends UnaryTransformer[String, Seq[String], Sequencer] with DefaultParamsWritable {
  
  final val diacritic: BooleanParam = new BooleanParam(this, "diacritic", "keep diacritics or not")
  final val phoneme: BooleanParam = new BooleanParam(this, "phoneme", "use phoneme")
  
  setDefault(diacritic -> true, phoneme -> true)
  
  def setDiacritic(value: Boolean): this.type = set(diacritic, value)
  def setPhoneme(value: Boolean): this.type = set(phoneme, value)
  
  def this() = this(Identifiable.randomUID("sequencer"))
  
  override protected def createTransformFunc: (String) => Seq[String] = {
    Sequencer.transform(_, $(diacritic), $(phoneme))
  }

  override protected def outputDataType = new ArrayType(StringType, true)
}


object Sequencer extends DefaultParamsReadable[Sequencer] {
  def transform(x: String, diacritic: Boolean, phoneme: Boolean = true): Seq[String] = {
    var x0 = VieMap.normalize(x)
    for (punctuation <- Lexicon.punctuations.keySet) {
      val pattern = Lexicon.punctuations(punctuation)
      x0 = x0.replaceAll(pattern, "~" + punctuation + "~")
    }
    val tps = VieMap.threeMaps
    val result = ListBuffer[String]()
    val tokens = x0.split("[~]+").filter(_.nonEmpty)
    for (j <- 0 until tokens.size) {
      val t = tokens(j)
      var token = t
      if (t.head == 'd' || t.head == 'đ') {
        result.append(t.head.toString)
        token = t.tail
      }
      if (t.toLowerCase.startsWith("qu")) {
        result.append(t.substring(0, 2))
        token = t.substring(2)
      }
      if (t.size >= 3 && t.toLowerCase.startsWith("gi") && tps._1.contains(t.charAt(2).toString)) {
        // giả, giảng, giấy, NOT gìn
        result.append(t.substring(0, 2))
        token = t.substring(2)
      }
      
      val triples = tps._3.keySet
      val v = triples.find(v => token.contains(v)).getOrElse("NA")
      if (v != "NA") {
        val j = token.indexOf(v)
        if (j >= 0) {
          val left = token.substring(0, j)
          if (left.nonEmpty) result.append(left)
          result.append(v)
          val right = token.substring(j+v.length)
          if (right.nonEmpty) result.append(right)
        }
      } else {
        val pairs = tps._2.keySet
        val v = pairs.find(v => token.contains(v)).getOrElse("NA")
        if (v != "NA") {
          val j = token.indexOf(v)
          if (j >= 0) {
            val left = token.substring(0, j)
            if (left.nonEmpty) result.append(left)
            result.append(v)
            val right = token.substring(j+v.length)
            if (right.nonEmpty) result.append(right)
          }
        } else {
          val singles = tps._1.keySet
          val v = singles.find(v => token.contains(v)).getOrElse("NA")
          if (v != "NA") {
            val j = token.indexOf(v)
            if (j >= 0) {
              val left = token.substring(0, j)
              if (left.nonEmpty) result.append(left)
              result.append(v)
              val right = token.substring(j+v.length)
              if (right.nonEmpty) result.append(right)
            }
          } else result.append(token) 
        }
      }
    }
    if (!diacritic)
      result.map(e => e.map(c => VieMap.diacritics.getOrElse(c, c)))
    else result.toList
  }

  override def load(path: String): Sequencer = super.load(path)
}