package vlp.vdg

import com.intel.analytics.bigdl.Module
import org.apache.spark.ml.PipelineModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.slf4j.LoggerFactory

/**
  * An abstract VDG model.
  * @param config
  */
abstract class M(config: ConfigVDG) extends Serializable {
  final val logger = LoggerFactory.getLogger(getClass.getName)

  /**
    * Builds pre-processor to transform a training data frame.
    * @param trainingSet
    * @return a pipeline model
    */
  def buildPreprocessor(trainingSet: DataFrame): PipelineModel

  /**
    * Builds the core DL model, which is a transducer.
    * @param inputSizes
    * @param outputSize
    * @return a sequential model.
    */
  def transducer(inputSizes: Array[Int], outputSize: Int): Module[Float]

  /**
    * Trains the transducer on a pair of training set and validation set.
    * @param trainingSet
    * @param validationSet
    * @return a module
    */
  def train(trainingSet: DataFrame, validationSet: DataFrame): Module[Float]

  /**
    * Predicts a new data set.
    * @param dataset
    * @param preprocessor
    * @param module
    * @return a RDD of Row objects. Each row contains 3 sequences (xs, ys, zs) where
    *         xs is the input, ys is the correct output sequences and zs is the prediction sequence.
    */
  def predict(dataset: DataFrame, preprocessor: PipelineModel, module: Module[Float]): RDD[Row]

  /**
    * Evaluates the performance of a model on a data set, don't count the space characters.
    * @param dataset
    * @param preprocessor
    * @param module
    */
  def eval(dataset: DataFrame, preprocessor: PipelineModel, module: Module[Float]) = {
    val result = test(dataset, preprocessor, module)
    // compute the total number of visible characters in the input sequences
    val numChars = result.map(row => row.getAs[Seq[String]](0).mkString.filterNot(_ == ' ').size).sum
    val rdd = result.map(row => {
      (row.getAs[Seq[String]](1).mkString.filterNot(_ == ' '), row.getAs[Seq[String]](2).mkString.filterNot(_ == ' '))
    })
    val (corrects, total) = rdd.map(pair => {
      var count = 0
      pair._1.zip(pair._2).foreach { case (a, b) =>
        if (a == b) count = count + 1
      }
      (count, pair._1.size)
    }).reduce { case (u, v) => (u._1 + v._1, u._2 + v._2) }
    logger.info(s"accuracy = $corrects/$total = ${corrects.toDouble/total}")
    if (total != numChars) { // M2 or M3
      (corrects + (numChars - total))/(numChars.toDouble)
    } else { // M1
      corrects/(total.toDouble)
    }
  }

  /**
    * Runs the model on a dataset.
    * @param dataset
    * @param preprocessor
    * @param module
    * @return a RDD of Row objects.
    */
  def test(dataset: DataFrame, preprocessor: PipelineModel, module: Module[Float]): RDD[Row] = {
    val result = predict(dataset, preprocessor, module)
    if (config.verbose) {
      result.take(10).foreach(row => {
        val xs = row.getAs[Seq[String]](0)
        val ys = row.getAs[Seq[String]](1)
        val zs = row.getAs[Seq[String]](2)
        logger.info("x=" + xs.mkString)
        logger.info("y=" + ys.mkString)
        logger.info("z=" + zs.mkString)
      })
    }
    result
  }

  /**
    * Restore diacritics of an input text.
    * @param text
    * @param preprocessor
    * @param module
    * @return a sentence.
    */
  def test(text: String, preprocessor: PipelineModel, module: Module[Float]): String
}
