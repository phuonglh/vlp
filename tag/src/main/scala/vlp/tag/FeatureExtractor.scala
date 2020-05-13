package vlp.tag

import org.apache.spark.SparkException
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{ArrayType, StringType, StructType}
import org.apache.spark.sql.functions._
import vlp.tok.WordShape

import scala.collection.mutable.ListBuffer

/**
  * Feature extractor for Conditional Markov Model.
  *
  * phuonglh, phuonglh@gmail.com
  *
  */

trait FeatureExtractorParams extends Params  {
  final val wordCol: Param[String] = new Param[String](this, "wordCol", "word column name")
  final val tagCol: Param[String] = new Param[String](this, "tagCol", "tag column name")
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  final val markovOrder: IntParam = new IntParam(this, "markovOrder", "Markov order", ParamValidators.gt(0))

  final val featureTypes: StringArrayParam = new StringArrayParam(this, "feature types", "activated feature types")

  final def getOutputCol: String = $(outputCol)
  final def getMarkovOrder: Int = $(markovOrder)
  final def getFeatureTypes: Array[String] = $(featureTypes)

  setDefault(markovOrder -> 1, wordCol -> "word", tagCol -> "tag", outputCol -> "f",
    featureTypes -> Array("currentWord", "previousWord", "nextWord", "currentShape", "nextShape", "previous2Word", "next2Word"))
}

class FeatureExtractor(override val uid: String) extends Transformer with FeatureExtractorParams with DefaultParamsWritable {

  final val BOS = "$BOS"
  final val EOS = "$EOS"
  final val PUNCT = "PUNCT"


  def this() = this(Identifiable.randomUID("featureExtractor"))

  def setWordCol(value: String): this.type = set(wordCol, value)
  def setTagCol(value: String): this.type = set(tagCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setMarkovOrder(value: Int): this.type = set(markovOrder, value)
  def setFeatureTypes(value: Array[String]): this.type = set(featureTypes, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val extractFunction = udf { (ws: Seq[String], ts: Seq[String]) =>
      FeatureExtractor.extract(ws, ts, $(featureTypes), $(markovOrder))
    }

   dataset.select(dataset.col("*"), extractFunction(dataset.col($(wordCol)), dataset.col($(tagCol))).as($(outputCol)))
  }

  override def copy(extra: ParamMap): FeatureExtractor = defaultCopy(extra)

  override def transformSchema(schema: StructType) = {
    SchemaUtils.checkColumnType(schema, $(wordCol), ArrayType(StringType, true))
    SchemaUtils.checkColumnType(schema, $(tagCol), ArrayType(StringType, true))
    SchemaUtils.appendColumn(schema, $(outputCol), StringType)
  }
}

object FeatureExtractor extends DefaultParamsReadable[FeatureExtractor] {

  final val BOS = "$BOS"
  final val EOS = "$EOS"
  final val PUNCT = "PUNCT"

  override def load(path: String): FeatureExtractor = super.load(path)

  def extract(ws: Seq[String], ts: Seq[String], featureTypes: Seq[String], markovOrder: Int): String = {
    val n = ws.size
    val data = new ListBuffer[(String, String, String)]
    for (j <- 0 until n) {
      val features = extract(ws, ts, featureTypes, markovOrder, j)
      data.append((ws(j), features.mkString(" "), ts(j)))
    }
    data.mkString(" ")
  }

  def extract(words: Seq[String], tags: Seq[String], featureTypes: Seq[String], markovOrder: Int, j: Int): Seq[String] = {
    val features = new ListBuffer[String]
    val n = words.size
    // word and shape features
    for (featureType <- featureTypes) {
      featureType match {
        case "previousWord" => features.append("pw=" + (if (j > 0) words(j-1) else BOS))
        case "currentWord" => features.append("cw=" + words(j))
        case "nextWord" => features.append("nw=" + (if (j < n-1) words(j+1) else EOS))
        case "currentShape" => {
          val shape = if (words(j) != PUNCT) WordShape.shape(words(j)) else PUNCT
          if (shape.nonEmpty && shape != "lower") features.append("cs=" + shape)
        }
        case "nextShape" => {
          if (j < n-1) {
            val shape = if (words(j+1) != PUNCT) WordShape.shape(words(j+1)) else PUNCT
            if (shape.nonEmpty && shape != "lower") features.append("ns=" + shape)
          }
        }
        case "previous2Word" => features.append("p2w=" + (if (j > 1) words(j-2) else BOS))
        case "next2Word" => features.append("n2w=" + (if (j < n-2) words(j+2) else EOS))
        case _ => throw new SparkException(s"FeatureExtractor does not support this feature type: " + featureType)
      }
    }
    // tag features
    markovOrder match {
      case 1 => features.append("pt=" + (if (j > 0) tags(j - 1) else BOS))
      case 2 => {
        val pt = (if (j > 0) tags(j - 1) else BOS)
        val at = (if (j > 1) tags(j - 2) else BOS)
        features.append("pt=" + pt)
        features.append("at=" + at)
        features.append("bt=" + at + ';' + pt)
      }
      case _ => throw new SparkException(s"Markov order " + markovOrder + " is not supported.")
    }
    features
  }
}
