package vlp.zoo

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.Shape

import com.intel.analytics.zoo.pipeline.api.keras.layers.Dense
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.layers.Select
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.layers.SelectTable
import vlp.tok.VietnameseTokenizer
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame

class BERT(sparkSession: SparkSession, config: ConfigSHINRA) {

  def train(df: DataFrame): Unit = {
    val clazzIndexer = new StringIndexer().setInputCol("clazz").setOutputCol("label").setHandleInvalid("error")
    val tokenizer = new VietnameseTokenizer().setInputCol("body").setOutputCol("tokens").setSplitSentences(true).setToLowercase(true).setConvertNumber(true).setConvertPunctuation(true)
    val remover = new StopWordsRemover().setInputCol("tokens").setOutputCol("words").setStopWords(Array("[num]", "punct"))
    val countVectorizer = new CountVectorizer().setInputCol("words").setOutputCol("features").setMinDF(config.minFrequency)
    val pipeline = new Pipeline().setStages(Array(clazzIndexer, tokenizer, remover, countVectorizer))
    val preprocessor = pipeline.fit(df)
    val alpha = preprocessor.transform(df)
    alpha.show()
  }

  def buildModel(vocabSize: Int, maxSeqLen: Int, hiddenSize: Int): Model[Float] = {
    val inputIds = Input[Float](inputShape = Shape(maxSeqLen))
    val segmentIds = Input[Float](inputShape = Shape(maxSeqLen))
    val positionIds = Input[Float](inputShape = Shape(maxSeqLen))
    val masks = Input[Float](inputShape = Shape(1, 1, maxSeqLen))
    val bert = com.intel.analytics.zoo.pipeline.api.keras.layers.BERT[Float](vocab = vocabSize, hiddenSize = hiddenSize, nBlock = 12, nHead = 12,
      maxPositionLen = maxSeqLen, intermediateSize = 3072, outputAllBlock = false)

    val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masks))
    val selectNode = SelectTable[Float](0).inputs(bertNode)
    val output = Dense[Float](hiddenSize, activation = "softmax").inputs(selectNode)
    val model = Model[Float](Array(inputIds, segmentIds, positionIds, masks), output)
    model
  }

}

object BERT {

  // def test(model: Model[Float]): Unit = {
  //   val maxSeqLen = 12
  //   val is = Tensor[Float](Array[Float](22, 7, 12, 9, 9, 15, 20, 10, 80, 23, 2, 85), Array(1, maxSeqLen))
  //   val ss = Tensor[Float](1, maxSeqLen).zero()
  //   val ps = Tensor[Float](1, maxSeqLen).zero()
  //   for (j <- 1 to maxSeqLen) {
  //     ps.setValue(1, j, j - 1.0f)
  //   }
  //   val ms = Tensor[Float](1, 1, 1, maxSeqLen).zero()
  //   ms.fill(1.0f)
  //   val input = T(is, ss, ps, ms)
  //   val model = buildModel(100, maxSeqLen, 24)
  //   println(input)
  //   val output = model.forward(input)
  //   println(output)
  // }

  def main(args: Array[String]): Unit = {
    val sparkConfig = Engine.createSparkConf()
      .setMaster("local[*]")
      .set("spark.executor.memory", "8g")
      .setAppName("zoo.BERT")
    val sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate()
    val sparkContext = sparkSession.sparkContext
    Engine.init

    val config = ConfigSHINRA()
    val app = new BERT(sparkSession, config)

    val df = sparkSession.read.json(config.dataPath + "1000.json")
    app.train(df)

    sparkSession.stop()
  }
}
