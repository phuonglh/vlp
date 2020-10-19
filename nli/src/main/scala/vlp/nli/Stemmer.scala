package vlp.nli

import org.tartarus.snowball.SnowballStemmer

import org.apache.spark.sql.types.{DataType, StringType, ArrayType}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.UnaryTransformer



class Stemmer(override val uid: String) extends UnaryTransformer[Seq[String], Seq[String], Stemmer] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("stemmer"))

  val language: Param[String] = new Param(this, "language", "stemming language (case insensitive).")
  def getLanguage: String = $(language)
  def setLanguage(value: String): this.type = set(language, value)

  setDefault(language -> "English")

  override protected def createTransformFunc: Seq[String] => Seq[String] = { strArray =>
    val stemClass = Class.forName("org.tartarus.snowball.ext." + $(language).toLowerCase + "Stemmer")
    val stemmer = stemClass.newInstance.asInstanceOf[SnowballStemmer]
    strArray.map(originStr => {
      stemmer.setCurrent(originStr)
      stemmer.stem()
      stemmer.getCurrent
    })
  }

  override protected def validateInputType(inputType: DataType): Unit = {
  }

  override protected def outputDataType: DataType = new ArrayType(StringType, false)

  override def copy(extra: ParamMap): Stemmer = defaultCopy(extra)
}

object Stemmer extends DefaultParamsReadable[Stemmer] {

  override def load(path: String): Stemmer = super.load(path)
}
