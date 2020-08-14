package vlp.ner

import org.apache.spark.ml.util.{DefaultParamsWritable, DefaultParamsReadable}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{StructField, StructType, ArrayType, StringType}
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCol}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{Dataset, DataFrame}


/**
  * A transformer which looks closely into an interested mention and its surrounding context. For example,
  * given xs = [I, am, living, in, HCM, City, from, 2015], and "ys = [O, O, O, O, ORG, ORG, O, O]", it extracts the 
  * subsequence ss = [HCM, City]. This helps extract a vocabulary of important words for bag-of-word representations. 
  * 
  * phuonglh@gmail.com
  * 
  * @param uid unique id of this transformer
  */
class MentionExtractor(override val uid: String) extends Transformer with DefaultParamsWritable with HasInputCols with HasOutputCol {

  def this() = this(Identifiable.randomUID("mentionExtr"))

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  private def createTransformFunc: (Seq[String], Seq[String]) => Seq[String] = {(xs: Seq[String], ys: Seq[String]) => 
    val us = xs.zip(ys)
    val vs = us.filter { case (x, y) => y != "o" }
    vs.map(_._1)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val transformUDF = udf(this.createTransformFunc, new ArrayType(StringType, false))
    dataset.withColumn($(outputCol), transformUDF(dataset($(inputCols)(0)), dataset($(inputCols)(1))))
  }

  override def copy(extra: ParamMap) = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputFields = schema.fields :+ StructField($(outputCol), new ArrayType(StringType, false), nullable = false)
    StructType(outputFields)
  }
}

object MentionExtractor extends DefaultParamsReadable[MentionExtractor] {
  override def load(path: String): MentionExtractor = super.load(path)
}

