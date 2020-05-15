package vlp.vdr

import org.apache.spark.SparkException
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset, RowFactory}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer

/**
  * phuonglh, 10/31/17, 20:29
  * 
  * Sample data from (x, y) sequences.
  */
class Sampler(override val uid: String) extends Transformer with DefaultParamsWritable {
  final val featureCol: Param[String] = new Param[String](this, "featureCol", "feature column name")
  final val labelCol: Param[String] = new Param[String](this, "labelCol", "label column name")
  final val featureTypes: StringArrayParam = new StringArrayParam(this, "feature types", "activated feature types")
  final val markovOrder: IntParam = new IntParam(this, "markovOrder", "Markov order")

  def setFeatureCol(value: String): this.type = set(featureCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setMarkovOrder(value: Int): this.type = set(markovOrder, value)
  
  def getFeatureTypes: Array[String] = $(featureTypes)
  def getMarkovOrder: Int = $(markovOrder)
  
  setDefault(markovOrder -> 1, featureTypes -> Array("c(-2)", "c(-1)", "c(0)", "c(+1)", "c(+2)", "w(-1)", "w(0)", "w(+1)", "joint"), 
    featureCol -> "f", labelCol -> "c")
  
  def this() = this(Identifiable.randomUID("sampler"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val input = dataset.select("x", "y").rdd
    val output = input.flatMap(row => {
      val x = row.getAs[Seq[String]](0)
      val y = row.getAs[Seq[String]](1)
      val data = Sampler.extract(x, y, $(featureTypes), $(markovOrder))
      data.map(datum => RowFactory.create(datum.features.mkString(" "), datum.label))
    })
    val schema = transformSchema(dataset.schema, logging = true)
    dataset.sparkSession.createDataFrame(output, schema)
  }

  override def copy(extra: ParamMap) = defaultCopy(extra)

  override def transformSchema(schema: StructType) = {
    StructType(Array(StructField($(featureCol), StringType, true), StructField($(labelCol), StringType, true)))
  }
}

object Sampler extends DefaultParamsReadable[Sampler] {
  final val logger = LoggerFactory.getLogger(getClass.getName)
  final val BOS = "$BOS"
  final val EOS = "$EOS"

  case class Datum(features: Seq[String], label: String)

  override def load(path: String): Sampler = super.load(path)
  
  def extract(x: Seq[String], y: Seq[String], featureTypes: Seq[String], markovOrder: Int): Seq[Datum] = {
    def trigger(label: String): Boolean = {
      VieMap.threeMaps._3.contains(label) || VieMap.threeMaps._2.contains(label) || VieMap.threeMaps._1.contains(label)
    }
    
    if (x.size != y.size) {
      logger.error(x.toList.toString())
      logger.error(y.toList.toString())
      throw new IllegalArgumentException("Word and tag sequence lengths do not match.")
    }
    val buffer = new ListBuffer[Datum]()
    for (j <- 0 until y.size)
      if (trigger(y(j))) {
        val fs = extract(x, y, featureTypes, markovOrder, j)
        buffer += Datum(fs, y(j).toString)
      }
    buffer  
  }
  
  def extract(x: Seq[String], y: Seq[String], featureTypes: Seq[String], markovOrder: Int, j: Int): Seq[String] = {
    val features = new ListBuffer[String]
    val n = x.size
    val cs = new ListBuffer[String]()
    for (featureType <- featureTypes) {
      featureType match {
        case "c(-2)" => {
          val f = (if (j > 1) x(j-2) else BOS)
          cs.append(f)
          features.append("c(-2)=" + f)
        }
        case "c(-1)" => {
          val f = (if (j > 0) x(j-1) else BOS)
          cs.append(f)
          features.append("c(-1)=" + f)
        }
        case "c(0)" => {
          cs.append(x(j))
          features.append("c(0)=" + x(j))
        }
        case "c(+1)" => {
          val f = (if (j < n-1) x(j+1) else EOS)
          cs.append(f)
          features.append("c(+1)=" + f)
        }
        case "c(+2)" => {
          val f = (if (j < n-2) x(j+2) else EOS)
          cs.append(f)
          features.append("c(+2)=" + f)
        }
        case "w(0)" => {
          var u = j
          var v = j
          while (u >= 0 && x(u) != " ") u = u - 1
          while (v < x.size && x(v) != " ") v = v + 1
          features.append("w(0)=" + x.slice(u, v).mkString("").trim)
        }
        case "w(-1)" => {
          var v = j
          while (v >= 0 && x(v) != " ") v = v - 1
          if (v > 0) {
            var u = v-1
            while (u >= 0 && x(u) != " ") u = u - 1
            features.append("w(-1)=" + x.slice(u, v).mkString("").trim)
          } else features.append("w(-1)=" + BOS)
        }
        case "w(+1)" => {
          var u = j
          while (u < x.size && x(u) != " ") u = u + 1
          if (u < x.size) {
            var v = u + 1
            while (v < x.size && x(v) != " ") v = v + 1
            features.append("w(+1)=" + x.slice(u, v).mkString("").trim)
          } else features.append("w(+1)=" + EOS)
        }
        case "joint" => {
          for (k <- 0 until cs.size - 1) 
            features.append(k.toString + (k+1).toString + '=' + cs(k) + '+' + cs(k+1))
          features.append("13=" + cs(1) + '+' + cs(3))
        }
        case _ => throw new SparkException("This feature type is not supported: " + featureType)
      }
    }
    markovOrder match {
      case 0 => {}
      case 1 => features.append("t(-1)=" + (if (j > 0) y(j - 1) else BOS))
      case 2 => {
        val t1 = if (j > 0) y(j - 1) else BOS
        val t2 = if (j > 1) y(j - 2) else BOS
        features.append("t(-1)=" + t1)
        features.append("t(-2)=" + t2)
        features.append("t(-2)t(-1)=" + t2 + t1)
      }
      case _ => throw new SparkException("This Markov order is not supported: " + markovOrder)
    }
    features
  }
  
}