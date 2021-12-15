package vlp.tdp

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, SparseVector}
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer

/**
  * Created by phuonglh on 7/14/17.
  *
  * Loads a trained pipeline model, builds a MLP classifier
  * and predicts the best transition for a given parsing context. This class serves
  * as the main facility for parallel evaluation of a test corpus.
  */

class MLP(spark: SparkSession, pipeline: PipelineModel, featureExtractor: FeatureExtractor) extends ANN(spark, pipeline, featureExtractor) {
  val logger = LoggerFactory.getLogger(getClass)

  val mlp = pipeline.stages(pipeline.stages.length - 1).asInstanceOf[MultilayerPerceptronClassificationModel]
  val (theta, b) = {
    val layers = mlp.getLayers
    val weights = mlp.weights.toArray
    val n = layers.size
    val theta = new ListBuffer[Array[Double]]
    val b = new ListBuffer[Array[Double]]
    var u = 0
    var v = 0
    for (i <- 0 until n-1) {
      v = u + layers(i) * layers(i+1)
      theta += weights.slice(u, v)
      b += weights.slice(v, v + layers(i + 1))
      u = v + layers(i + 1)
    }
    (theta.toList, b.toList)
  }

  val broadcastDict: Option[Broadcast[Map[String, Int]]] = {
    val dict = countVectorizer.vocabulary.zipWithIndex.toMap
    Some(spark.sparkContext.broadcast(dict))
  }


  override def info(): String = {
    val sb = new StringBuilder(super.info())
    for (i <- 0 until theta.size) {
      sb.append(s"length(theta($i)) = " + theta(i).length + "\n")
      sb.append(s"theta($i) = [" + theta(i).take(20).mkString(", ") + ",...]\n")
      sb.append(s"beta($i) = [" + b(i).mkString(", ") + "]\n")
    }
    sb.toString()
  }
  
  /**
    * Computes a forward pass through a layer of the MLP with (w, b) coefficients and input vector x.
    * @param w weight vector of the layer
    * @param rows number of rows of the weight matrix (of size numOut)
    * @param b bias vector (of size numOut)
    * @param x input vector
    * @param lastLayer if this is the computation in the last layer.         
    * @return resulting activation values
    */
  private def forward(w: Array[Double], rows: Int, b: Array[Double], x: Array[Double], lastLayer: Boolean = true): Array[Double] = {
    def sigmoid: (Double) => Double = x => 1.0 / (1 + math.exp(-x))
    
    // afin layer: z = w*x + b
    val theta = new DenseMatrix(rows, x.length, w)
    val v = theta.multiply(new DenseVector(x)).values
    val z = v.zip(b).map(p => p._1 + p._2)
    // sigmoid or softmax layer: f(z)
    if (lastLayer) z else z.map(sigmoid)
  }

  def predict(features: String): List[String] = {
    val dictBr = broadcastDict.get
    val jj = features.toLowerCase.split("\\s+").map {
      f => dictBr.value.get(f) match {
        case Some(j) => j
        case None => -1
      }
    }.filter(j => j >= 0).sorted.distinct
    val values = jj.map(_ => 1.0)

    val layers = mlp.getLayers

    // compute a forward pass through the network
    var x = new SparseVector(layers.head, jj, values).toArray
    for (i <- 0 until (layers.length - 1)) {
      if (i < layers.length - 2)
        x = forward(theta(i), layers(i + 1), b(i), x, false)
      else
        x = forward(theta(i), layers(i + 1), b(i), x)
    }

    val pairs = x.zipWithIndex
    // take the two best labels
    pairs.sortBy(_._1).map(_._2).reverse.take(2).toList.map(i => transitions(i))
  }
}
