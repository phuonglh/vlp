package vlp.zoo

import org.apache.spark.sql.SparkSession
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models._
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.optim.ValidationMethod
import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.zoo.feature.text.TextSet
import com.intel.analytics.bigdl.optim.ValidationResult

/**
  * Transformer-based model for text classification.
  * phuonglh, July 2020.
  * 
  */

class TransformerClassifier[T: ClassTag](classNum: Int, embeddingDimension: Int, sequenceLength: Int = 500, embedding: Embedding[T])(implicit ev: TensorNumeric[T]) 
  extends ZooModel[Activity, Activity, T] {

  override def buildModel(): AbstractModule[Activity, Activity, T] = {
    val model = Sequential[T]()
    model.add(embedding)
    // TODO.
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(Dense(classNum, activation = "softmax"))
    model
  }

  // For the following methods, please refer to KerasNet for documentation.
  def compile(optimizer: OptimMethod[T], loss: Criterion[T], metrics: List[ValidationMethod[T]] = null)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].compile(optimizer, loss, metrics)
  }

  def fit(x: TextSet, batchSize: Int, nbEpoch: Int, validationData: TextSet = null)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].fit(x, batchSize, nbEpoch, validationData)
  }

  def evaluate(x: TextSet, batchSize: Int)(implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    model.asInstanceOf[KerasNet[T]].evaluate(x, batchSize)
  }

  def predict(x: TextSet, batchPerThread: Int): TextSet = {
    model.asInstanceOf[KerasNet[T]].predict(x, batchPerThread)
  }

  def setTensorBoard(logDir: String, appName: String): Unit = {
    model.asInstanceOf[KerasNet[T]].setTensorBoard(logDir, appName)
  }

  def setCheckpoint(path: String, overWrite: Boolean = true): Unit = {
    model.asInstanceOf[KerasNet[T]].setCheckpoint(path, overWrite)
  }
  
}

object TransformerClassifier {

 def apply[@specialized(Float, Double) T: ClassTag](classNum: Int, embeddingFile: String, wordIndex: Map[String, Int] = null, sequenceLength: Int = 500)
  (implicit ev: TensorNumeric[T]): TransformerClassifier[T] = {
    val embedding = WordEmbedding(embeddingFile, wordIndex, inputLength = sequenceLength)
    new TransformerClassifier[T](classNum, embedding.outputDim, sequenceLength, embedding).build()
  }

  /**
   * Load an existing TransformerClassifier model (with weights).
   *
   * @param path The path for the pre-defined model.
   *             Local file system, HDFS and Amazon S3 are supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx".
   *             Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path for pre-trained weights if any. Default is null.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   */
  def loadModel[T: ClassTag](path: String, weightPath: String = null)(implicit ev: TensorNumeric[T]): TransformerClassifier[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[TransformerClassifier[T]]
  }

}
