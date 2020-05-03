package vlp.tag

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, RowFactory}

/**
  * phuonglh
  *
  * Extractor of samples for training CMM models in part-of-speech tagger.
  */

class SampleExtractor(override val uid: String) extends Transformer with DefaultParamsWritable {
  
  def this() = this(Identifiable.randomUID("sampleExtractor"))
  
  final val sampleCol: Param[String] = new Param[String](this, "sampleCol", "input column name")
  final val tokenCol:  Param[String] = new Param[String](this, "tokenCol", "token column name")
  final val featureCol: Param[String] = new Param[String](this, "featureCol", "feature column name")
  final val labelCol: Param[String] = new Param[String](this, "labelCol", "label column name")

  def setSampleCol(value: String): this.type = set(sampleCol, value)
  def setTokenCol(value: String): this.type = set(tokenCol, value)
  def setFeatureCol(value: String): this.type = set(featureCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  private case class Datum(features: String, label: String)
  
  override def transform(dataset: Dataset[_]): DataFrame = {
    val input = dataset.select($(sampleCol)).rdd
    val pattern = """\(.*?\)""".r
    val output = input.flatMap(row => {
      val sample = pattern.findAllIn(row.getString(0)).map(e => e.substring(1, e.size-1)).toList
      sample.map(e => e.split(",")).map(a => RowFactory.create(a(0), a(1), a(2)))
    })
    transformSchema(dataset.schema, logging = true)
    val schema = StructType(Array(StructField($(tokenCol), StringType, true), StructField($(featureCol), StringType, true), StructField($(labelCol), StringType, true)))
    dataset.sparkSession.createDataFrame(output, schema)
  } 

  override def copy(extra: ParamMap) = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    StructType(Array(StructField($(tokenCol), StringType, true),
      StructField($(featureCol), StringType, true), 
      StructField($(labelCol), StringType, true)))
  }
}

object SampleExtractor extends DefaultParamsReadable[SampleExtractor] {
  override def load(path: String): SampleExtractor = super.load(path)
}