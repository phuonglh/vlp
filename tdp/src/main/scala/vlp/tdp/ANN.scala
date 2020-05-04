package vlp.tdp

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{CountVectorizerModel, StringIndexerModel}
import org.apache.spark.sql.SparkSession

/**
  * Created by phuonglh on 7/15/17.
  * 
  * Base class for both [[MLR]] and [[MLP]]
  */
abstract class ANN(val spark: SparkSession, val pipeline: PipelineModel, val featureExtractor: FeatureExtractor) extends Serializable {
  protected val transitions = pipeline.stages(0).asInstanceOf[StringIndexerModel].labels
  protected val countVectorizer = pipeline.stages(2).asInstanceOf[CountVectorizerModel]

  /**
    * Gets some information about the model.
    * @return a string
    */
  def info(): String = {
    val sb = new StringBuilder()
    sb.append("\n#(transitions) = " + transitions.length)
    sb.append(", [")
    sb.append(transitions.mkString(", "))
    sb.append("]")
    sb.append("\n#(vocab) = " + countVectorizer.vocabulary.length + "\n")
    sb.toString()
  }
  
  def predict(features: String): List[String]

  /**
    * Predicts the best transitions for a given parsing config.
    * @param config parsing config
    * @return the best transitions
    */
  def predict(config: Config): List[String] = {
    val features = featureExtractor.extract(config)
    predict(features)
  }
  
}
