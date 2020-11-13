package vlp.vdg

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{PaddingParam, Sample}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.layers.Dense
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Reshape => ReshapeZoo}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Select => SelectZoo}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{SelectTable => SelectTableZoo}

import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.Shape


import scala.collection.mutable

/**
  * phuonglh
  *
  * Vietnamese Diacritic Generation: the character-based transformer model.
  */
class M4(config: ConfigVDG) extends M1(config) {
  final val paddingXs = PaddingParam[Float](Some(Array(Tensor(T(1f)), Tensor(T(0f)), Tensor(T(0f)), Tensor(T(1f)))))

    /**
   * Constructs a BERT model for VDG using Zoo layers
   **/ 
  override def transducer(inputSizes: Array[Int], outputSize: Int): Model[Float] = {
    val maxSeqLen = config.maxSequenceLength
    val inputIds = Input(inputShape = Shape(maxSeqLen))
    val segmentIds = Input(inputShape = Shape(maxSeqLen))
    val positionIds = Input(inputShape = Shape(maxSeqLen))
    val masks = Input(inputShape = Shape(maxSeqLen))
    val masksReshaped = ReshapeZoo(Array(1, 1, maxSeqLen)).inputs(masks)

    val bert = com.intel.analytics.zoo.pipeline.api.keras.layers.BERT(vocab = inputSizes(0), hiddenSize = config.encoderOutputSize, nBlock = config.numBlocks, nHead = config.numHeads,
      maxPositionLen = maxSeqLen, intermediateSize = config.hiddenUnits, outputAllBlock = false)

    val bertNode = bert.inputs(Array(inputIds, segmentIds, positionIds, masksReshaped))
    val bertOutput = SelectTableZoo(0).inputs(bertNode)

    val dense = Dense(outputSize).inputs(bertOutput)
    val output = com.intel.analytics.bigdl.nn.keras.SoftMax().inputs(dense)
    val model = Model(Array(inputIds, segmentIds, positionIds, masks), output)
    model
  }

  
  override def train(trainingSet: DataFrame, validationSet: DataFrame): Module[Float] = {
    val preprocessor = buildPreprocessor(trainingSet)
    logger.info("Saving the Spark pre-processing pipeline...")
    val modelSt = "M" + config.modelType + "X" + config.numHeads + "B" + config.numBlocks + "O" + config.encoderOutputSize + "H" + config.hiddenUnits
    val path = config.modelPath + modelSt + "/"
    preprocessor.write.overwrite().save(path)

    val inputLabels = preprocessor.stages(4).asInstanceOf[CountVectorizerModel].vocabulary
    val outputLabels = preprocessor.stages(5).asInstanceOf[CountVectorizerModel].vocabulary
    val inputEncoder = new TokenEncoder(inputLabels).setInputCol("x").setOutputCol("input")
      .setNumFeatures(inputLabels.length)
      .setSequenceLength(config.maxSequenceLength)
    val outputEncoder = new TokenEncoder(outputLabels).setInputCol("y").setOutputCol("output")
      .setNumFeatures(outputLabels.length)
      .setSequenceLength(config.maxSequenceLength)
    
    val numInputLabels = inputLabels.size
    val numOutputLabels = outputLabels.size
    logger.info("numInputLabels = " + numInputLabels)
    logger.info(inputLabels.mkString(" "))
    logger.info("numOutputLabels = " + numOutputLabels)
    logger.info(outputLabels.mkString(" "))
    
    val df0 = preprocessor.transform(trainingSet)
    val df1 = inputEncoder.transform(df0)
    val training = outputEncoder.transform(df1)
    training.show(10)

    val n = config.maxSequenceLength
    val trainingRDD = training.select("input", "output").rdd.map { row =>
      val y = row.get(1).asInstanceOf[Seq[Int]].toArray.map(e => e.toFloat + 1)
      val x = row.get(0).asInstanceOf[Seq[Int]].toArray.map(e => e.toFloat + 1)
      val tokenIds = Tensor(x, Array(n))
      val segmentIds = Tensor(n).fill(0f)
      val positionIds = Tensor((0 until n).toArray.map(_.toFloat), Array(n))
      val masks = Tensor(n).fill(1f)
      Sample(Array(tokenIds, segmentIds, positionIds, masks), Tensor(y, Array(n))) 
    }

    val df0v = preprocessor.transform(validationSet)
    val df1v = inputEncoder.transform(df0v)
    val validation = outputEncoder.transform(df1v)

    val validationRDD = validation.select("input", "output").rdd.map { row =>
      val y = row.get(1).asInstanceOf[Seq[Int]].toArray.map(e => e.toFloat + 1)
      val x = row.get(0).asInstanceOf[Seq[Int]].toArray.map(e => e.toFloat + 1)
      val tokenIds = Tensor(x, Array(n))
      val segmentIds = Tensor(n).fill(0f)
      val positionIds = Tensor((0 until n).toArray.map(_.toFloat), Array(n))
      val masks = Tensor(n).fill(1f)
      Sample(Array(tokenIds, segmentIds, positionIds, masks), Tensor(y, Array(n))) 
    }

    val model = transducer(Array(numInputLabels + 1), numOutputLabels)
    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainingRDD,
      criterion = TimeDistributedCriterion(ClassNLLCriterion[Float](paddingValue = 0), true),
      batchSize = config.batchSize,
      featurePaddingParam = paddingXs,
      labelPaddingParam = paddingY
    )

    val trainSummary = TrainSummary(appName = modelSt, logDir = "/tmp/vdg/summary/")
    val validationSummary = ValidationSummary(appName = modelSt, logDir = "/tmp/vdg/summary/")

    logger.info("Training a transformer model...")
    optimizer.setOptimMethod(new Adam[Float](learningRate = config.learningRate))
      .setEndWhen(Trigger.maxEpoch(config.epochs))
      .setValidation(Trigger.everyEpoch, validationRDD, Array(new TimeDistributedTop1Accuracy(paddingValue = 0)), config.batchSize)
      .setValidationSummary(validationSummary)
      .setTrainSummary(trainSummary)
      .optimize()
    val trainLoss = trainSummary.readScalar("Loss")
    val trainAccuracy = trainSummary.readScalar("TimeDistributedTop1Accuracy")
    val validationLoss = validationSummary.readScalar("Loss")
    val validationAccuracy = validationSummary.readScalar("TimeDistributedTop1Accuracy")
    logger.info("     Train Accuracy: " + trainAccuracy.mkString(", "))
    logger.info("Validation Accuracy: " + validationAccuracy.mkString(", "))
    logger.info("Saving the Transformer transducer...")
    model.saveModule(path + "vdg.bigdl", path + "vdg.bin", true)
  }

  override def predict(dataset: DataFrame, preprocessor: PipelineModel, module: Module[Float]): RDD[Row] = {
    val inputLabels = preprocessor.stages(4).asInstanceOf[CountVectorizerModel].vocabulary
    val outputLabels = preprocessor.stages(5).asInstanceOf[CountVectorizerModel].vocabulary

    val inputEncoder = new TokenEncoder(inputLabels).setInputCol("x").setOutputCol("input")
      .setNumFeatures(inputLabels.length)
      .setSequenceLength(config.maxSequenceLength)

    val df0 = preprocessor.transform(dataset)
    val df1 = inputEncoder.transform(df0) 

    val n = config.maxSequenceLength

    val rdd = df1.select("input").rdd.map { row =>
      val x = row.get(0).asInstanceOf[Seq[Int]].toArray.map(e => e.toFloat + 1)
      val tokenIds = Tensor(x, Array(n))
      val segmentIds = Tensor(n).fill(0f)
      val positionIds = Tensor((0 until n).toArray.map(_.toFloat), Array(n))
      val masks = Tensor(n).fill(1f)
      Sample(Array(tokenIds, segmentIds, positionIds, masks)) 
    }
    
    val predictions = module.predict(rdd).map(activity => {
      val ys = activity.toTensor[Float].split(1).toSeq
      val zs = ys.map(tensor => {
        val k = (0 until tensor.toArray().size).zip(tensor.toArray()).maxBy(p => p._2)._1
        outputLabels(k)
      })
      zs
    })
    df0.select("x", "y").rdd.zip(predictions).map(p => {
      val xs = p._1.getAs[Seq[String]](0)
      val ys = p._1.getAs[Seq[String]](1)
      val zs = p._2
      val n = Math.min(xs.size, config.maxSequenceLength)
      Row(xs.slice(0, n), ys.slice(0, n), zs.slice(0, n))
    })
  }

}

