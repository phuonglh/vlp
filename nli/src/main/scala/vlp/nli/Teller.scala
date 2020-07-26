package vlp.nli

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.Module
import org.apache.spark.ml.PipelineModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.slf4j.LoggerFactory
import com.intel.analytics.bigdl.nn.keras.{Sequential, Embedding, Dense}
import com.intel.analytics.bigdl.nn.ParallelTable
import org.apache.spark.sql.SparkSession
import com.intel.analytics.bigdl.nn.Linear
import com.intel.analytics.bigdl.nn.ReLU
import com.intel.analytics.bigdl.nn.ConcatTable
import com.intel.analytics.bigdl.nn.SoftMax
import com.intel.analytics.bigdl.utils.Shape
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.VectorAssembler


class Teller(sparkSession: SparkSession, config: ConfigTeller) {

  def preprocessor(input: DataFrame): Pipeline = {
    val premiseTokenizer = new Tokenizer().setInputCol("premise").setOutputCol("ps")
    val hypothesisTokenizer = new Tokenizer().setInputCol("hypothesis").setOutputCol("hs")
    val countVectorizer = new CountVectorizer().setInputCol("ps").setOutputCol("us").setMinTF(config.minFrequency).setBinary(true)
    VectorAssembler
  }

  /**
    * Builds the core DL model, which is a transducer.
    */
  def transducer(): Module[Float] = {
    val model = Sequential()
    val branches = ParallelTable()
    val sourceEmbedding = Embedding(config.numFeatures, config.embeddingSize, inputShape = Shape(config.maxSequenceLength))
    val source = Sequential().add(sourceEmbedding).add(Linear(config.embeddingSize, config.outputSize)).add(ReLU())
    val targetEmbedding = Embedding(config.numFeatures, config.embeddingSize, inputShape = Shape(config.maxSequenceLength))
    val target = Sequential().add(Linear(config.embeddingSize, config.outputSize)).add(ReLU())
    branches.add(source).add(target)
    model.add(branches).add(ConcatTable()).add(Dense(config.numLabels, activation = "softmax"))
  }

}