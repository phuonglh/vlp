package vlp.nli

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataType, ArrayType, StringType}

/**
  * A concept lookup which transforms a sequence of words into a sequence of related concepts
  * using a concept map. A default filter is also used to accept only interested relations that come 
  * from some specified languages.
  * 
  * phuonglh@gmail.com
  */
class ConceptLookup(val uid: String, val dictionary: Map[String, Set[String]], val filterFunc: String => Boolean) extends UnaryTransformer[Seq[String], Seq[String], ConceptLookup] with DefaultParamsWritable {

  var dictionaryBr: Option[Broadcast[Map[String, Set[String]]]] = None

  def this(dictionary: Map[String, Set[String]], filterFunc: String => Boolean) = {
    this(Identifiable.randomUID("lookup"), dictionary, filterFunc)
    val sparkContext = SparkSession.getActiveSession.get.sparkContext
    dictionaryBr = Some(sparkContext.broadcast(dictionary))
  }

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    def f(words: Seq[String]): Seq[String] = {
      val dict = dictionaryBr.get.value
      words.flatMap { x => 
        val concepts = dict.getOrElse(x, Set.empty)
        concepts.filter(filterFunc).filter(concept => !concept.contains(x)).toSeq
      }
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(StringType, false)
}

object ConceptLookup extends DefaultParamsReadable[ConceptLookup] {
  override def load(path: String): ConceptLookup = super.load(path)
}
