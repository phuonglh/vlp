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

import scala.collection.mutable

/**
  * phuonglh, 10/13/18, 10:13
  *
  * Vietnamese Diacritic Generation: the character-based model.
  */
class M1(config: ConfigVDG) extends M(config) {
  final val paddingX = PaddingParam[Float](Some(Array(Tensor(T(1f)))))
  final val paddingY = PaddingParam[Float](Some(Array(Tensor(T(0f)))))

  override def buildPreprocessor(trainingSet: DataFrame): PipelineModel = {
    val remover = new DiacriticRemover().setInputCol("text").setOutputCol("x0")
    val inputTokenizer = new RegexTokenizer().setInputCol("x0").setOutputCol("x").setPattern(".").setGaps(false).setToLowercase(true)
    val outputTokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("y0").setPattern(".").setGaps(false).setToLowercase(true)
    val outputConverter = new CharConverter().setInputCol("y0").setOutputCol("y")
    val inputVectorizer = new CountVectorizer().setInputCol("x").setMinDF(config.minFrequency).setBinary(true)
    val outputVectorizer = new CountVectorizer().setInputCol("y").setMinDF(config.minFrequency).setBinary(true)
    new Pipeline().setStages(Array(remover, inputTokenizer, outputTokenizer, outputConverter, inputVectorizer, outputVectorizer)).fit(trainingSet)
  }
  
  override def transducer(inputSizes: Array[Int], outputSize: Int): Module[Float] = {
    val model = Sequential[Float]()
    val merge = JoinTable[Float](2, 2).asInstanceOf[AbstractModule[Table, Tensor[Float], Float]]
    val first = BiRecurrent[Float](merge)
    val numInputLabels = inputSizes(0)
    if (config.gru)
      first.add(GRU(numInputLabels, config.hiddenUnits, config.dropout))
    else {
      if (config.peephole)
        first.add(LSTMPeephole(numInputLabels, config.hiddenUnits, config.dropout))
      else first.add(LSTM(numInputLabels, config.hiddenUnits, config.dropout))
    }
    val others = (1 until config.layers).map(_ => {
      val merge = JoinTable[Float](2, 2).asInstanceOf[AbstractModule[Table, Tensor[Float], Float]]
        val other = BiRecurrent[Float](merge)
        if (config.gru)
          other.add(GRU(2*config.hiddenUnits, config.hiddenUnits, config.dropout))
        else {
          if (config.peephole)
            other.add(LSTMPeephole(2*config.hiddenUnits, config.hiddenUnits, config.dropout))
          else other.add(LSTM(2*config.hiddenUnits, config.hiddenUnits, config.dropout))
        }
    })

    model.add(first)
    if (others.nonEmpty)
      others.foreach(model.add(_))
    model.add(TimeDistributed(Linear(2*config.hiddenUnits, outputSize)))
      .add(TimeDistributed(LogSoftMax()))
    model
  }
  
  override def train(trainingSet: DataFrame, validationSet: DataFrame): Module[Float] = {
    val preprocessor = buildPreprocessor(trainingSet)
    logger.info("Saving the Spark pre-processing pipeline...")
    val modelSt = "M" + config.modelType + (if (config.gru) "G"; else "L") + config.layers + "H" + config.hiddenUnits
    val path = config.modelPath + s"${modelSt}/"
    preprocessor.write.overwrite().save(path)

    val inputLabels = preprocessor.stages(4).asInstanceOf[CountVectorizerModel].vocabulary
    val outputLabels = preprocessor.stages(5).asInstanceOf[CountVectorizerModel].vocabulary
    val inputEncoder = new OneHotEncoder(inputLabels).setInputCol("x").setOutputCol("input")
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

    val trainingRDD = training.select("input", "output").rdd.map { row =>
      val y = row.get(1).asInstanceOf[Seq[Int]].toArray.map(e => e.toFloat + 1)
      val x = row.get(0).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Float]]].array.map(_.toArray)
      Sample(featureTensor = Tensor(x.flatten, Array(config.maxSequenceLength, numInputLabels)),
        labelTensor = Tensor(y, Array(config.maxSequenceLength))) 
    }

    val df0v = preprocessor.transform(validationSet)
    val df1v = inputEncoder.transform(df0v)
    val validation = outputEncoder.transform(df1v)

    val validationRDD = validation.select("input", "output").rdd.map { row =>
      val y = row.get(1).asInstanceOf[Seq[Int]].toArray.map(e => e.toFloat + 1)
      val x = row.get(0).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Float]]].array.map(_.toArray)
      Sample(featureTensor = Tensor(x.flatten, Array(config.maxSequenceLength, numInputLabels)),
        labelTensor = Tensor(y, Array(config.maxSequenceLength)))
    }

    val model = transducer(Array(numInputLabels), numOutputLabels)
    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainingRDD,
      criterion = TimeDistributedCriterion(ClassNLLCriterion[Float](paddingValue = 0), true),
      batchSize = config.batchSize,
      featurePaddingParam = paddingX,
      labelPaddingParam = paddingY
    )

    val trainSummary = TrainSummary(appName = modelSt, logDir = "dat/vdg/summary/")
    val validationSummary = ValidationSummary(appName = modelSt, logDir = "dat/vdg/summary/")

    logger.info("Training a RNN transducer model...")
    optimizer.setOptimMethod(new Adagrad[Float](learningRate = config.learningRate, learningRateDecay = 1E-3))
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
    logger.info("Saving the RNN transducer...")
    model.saveModule(path + "vdg.bigdl", path + "vdg.bin", true)
  }

  override def predict(dataset: DataFrame, preprocessor: PipelineModel, module: Module[Float]): RDD[Row] = {
    val inputLabels = preprocessor.stages(4).asInstanceOf[CountVectorizerModel].vocabulary
    val outputLabels = preprocessor.stages(5).asInstanceOf[CountVectorizerModel].vocabulary

    val inputEncoder = new OneHotEncoder(inputLabels).setInputCol("x").setOutputCol("input")
      .setNumFeatures(inputLabels.length)
      .setSequenceLength(config.maxSequenceLength)

    val df0 = preprocessor.transform(dataset)
    val df1 = inputEncoder.transform(df0) 

    val rdd = df1.select("input").rdd.map { row =>
      val x = row.get(0).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Float]]].array.map(_.toArray)
      Sample(featureTensor = Tensor(x.flatten, Array(config.maxSequenceLength, inputLabels.size)))
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

  /**
    * Restore diacritics of an input text.
    *
    * @param text
    * @param preprocessor
    * @param module
    * @return a sentence.
    */
  override def test(text: String, preprocessor: PipelineModel, module: Module[Float]): String = {
    val lineSlicer = new LineSlicer(config.maxSequenceLength)
    import scala.collection.JavaConversions._
    val slices = lineSlicer.split(text)
    val sparkSession = SparkSession.builder().getOrCreate()
    import sparkSession.implicits._
    val input = sparkSession.sparkContext.parallelize(slices).toDF("text")
    val output = test(input, preprocessor, module)
    output.map(row => row.getAs[Seq[String]](1).mkString).collect().mkString("\n")
  }
  
}

