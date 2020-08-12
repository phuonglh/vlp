package vlp.zoo

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}
import org.slf4j.LoggerFactory

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, Top1Accuracy, Top5Accuracy}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.Shape

import com.intel.analytics.zoo.pipeline.api.keras.layers.Dense
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.layers.Select
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.layers.InputLayer
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.layers.SelectTable

/**
 * BERT-based implementation for text classification.
 * phuonglh, <phuonglh@gmail.com>
 * August, 2020
 * 
*/ 

class BERT(sparkSession: SparkSession) {
}

object BERT {

  def buildModel(vocabSize: Int, maxSeqLen: Int, hiddenSize: Int): Model[Float] = {
    val inputIds = Input(inputShape = Shape(maxSeqLen))
    val segmentIds = Input(inputShape = Shape(maxSeqLen))
    val positionIds = Input(inputShape = Shape(maxSeqLen))
    val masks = Input(inputShape = Shape(1, 1, maxSeqLen))
    val bert = com.intel.analytics.zoo.pipeline.api.keras.layers.BERT(vocab = vocabSize, hiddenSize = hiddenSize, nBlock = 12, nHead = 12,
      maxPositionLen = maxSeqLen, intermediateSize = 1024, outputAllBlock = false)

    val shape = Shape(List(Shape(maxSeqLen), Shape(maxSeqLen), Shape(maxSeqLen), Shape(1, 1, maxSeqLen)))
    bert.build(KerasUtils.addBatch(shape))
    val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masks))
    val selectNode = SelectTable(0).inputs(bertNode)
    val output = Dense(hiddenSize, activation = "softmax").inputs(selectNode)
    val model = Model(Array(inputIds, segmentIds, positionIds, masks), output)
    model
  }

  def main(args: Array[String]): Unit = {
    val sparkConfig = Engine.createSparkConf()
      .setMaster("local[1]")
      .set("spark.executor.memory", "4g")
      .setAppName("zoo.BERT")
    val sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate()
    val sparkContext = sparkSession.sparkContext
    Engine.init
    val maxSeqLen = 12
    val is = Tensor(Array(22, 7, 12, 9, 9, 15, 20, 10, 80, 23, 2, 85), Array(maxSeqLen))
    val ss = Tensor(maxSeqLen).zero()
    val ps = Tensor(maxSeqLen).zero()
    for (j <- 1 to maxSeqLen) {
      ps.setValue(j, j - 1.0f)
    }
    val ms = Tensor(1, 1, maxSeqLen).zero()
    ms.fill(1.0f)
    val input = T(is, ss, ps, ms)
    val model = buildModel(100, maxSeqLen, 24)
    println(input)
    val output = model.forward(input)
    println(output)
  }
}
