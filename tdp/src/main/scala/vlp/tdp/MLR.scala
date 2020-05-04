package vlp.tdp

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

/**
  * Created by phuonglh on 7/2/17.
  * 
  * Loads a trained pipeline model, builds a logistic regression classifier 
  * and predicts the best transition for a given parsing context. This class serves 
  * as the main facility for parallel evaluation of a test corpus.
  */

class MLR(spark: SparkSession, pipeline: PipelineModel, featureExtractor: FeatureExtractor) extends ANN(spark, pipeline, featureExtractor) {
  
  val logger = LoggerFactory.getLogger(getClass)
  val (weights, bias) = {
    val numStages = pipeline.stages.length
    val classifier = pipeline.stages(numStages - 1).asInstanceOf[LogisticRegressionModel]
    logger.info("shape(theta) = " + classifier.coefficientMatrix.numRows + " x " + classifier.coefficientMatrix.numCols)
    logger.info("length(bias) = " + classifier.interceptVector.size)
    (classifier.coefficientMatrix, classifier.interceptVector)
  }
  
  val broadcastDict: Option[Broadcast[Map[String, Int]]] = {
    val dict = countVectorizer.vocabulary.zipWithIndex.toMap
    Some(spark.sparkContext.broadcast(dict))
  }

  /**
    * Predicts the best transitions for a given parsing config features.
    * @param features space-separated feature strings
    * @return the best transitions
    */
  def predict(features: String): List[String] = {
    val dictBr = broadcastDict.get
    val x = features.toLowerCase.split("\\s+").map {
      f => dictBr.value.get(f) match {
        case Some(j) => j
        case None => -1
      }
    }.filter(j => j >= 0).distinct
    val scores = (0 until transitions.length).map {
      k => {
        (k, bias(k) + x.map(j => weights(k, j)).sum)
      }
    }
    val (label, maxScore) = scores.maxBy(_._2)
    val ii = scores.filter(_._2 == maxScore).map(_._1)
    if (ii.length > 1) {
      ii.map(i => transitions(i)).toList
    } else List[String](transitions(label))
  }
  
}
