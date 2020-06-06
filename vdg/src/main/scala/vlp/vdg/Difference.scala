package vlp.vdg

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

/**
  * Detector of the output label, which compares the difference between input and output tokens.
  * @param uid
  */
class Difference(override val uid: String) extends UnaryTransformer[Seq[String], Seq[String], Difference] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("diff"))

  override protected def createTransformFunc: Seq[String] => Seq[String] = {
    def f(xs: Seq[String]): Seq[String] = {
      xs.map(x => {
        val y = DiacriticRemover.run(x)
        Difference.detect(y, x)
      })
    }

    f(_)
  }

  override protected def outputDataType: DataType = ArrayType(StringType, false)
}

object Difference extends DefaultParamsReadable[Difference] {

  /**
    * Detects the difference part between input x and output y, and returns
    * the different part of y (the label)
    * @param x
    * @param y
    * @return the difference part between x and y.
    */
  def detect(x: String, y: String): String = {
    assert(x.size == y.size)
    val js = x.zip(y).map(p => if (p._1 == p._2) 0; else 1)
      .zipWithIndex.filter(_._1 == 1).map(_._2)
    if (js.nonEmpty)
      y.substring(js.head, js.last + 1)
    else "S"
  }

  override def load(path: String): Difference = super.load(path)
}