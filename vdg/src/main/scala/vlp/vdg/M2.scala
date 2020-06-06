package vlp.vdg

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{PaddingParam, Sample}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.optim.{Adagrad, Optimizer, Predictor, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
  * Second model of VDG.
  *
  */
class M2(config: ConfigVDG) extends M(config) {
  val paddingX = PaddingParam[Float](Some(Array(Tensor(T(1f)))))
  final val paddingY = PaddingParam[Float](Some(Array(Tensor(T(0f)))))

  override def buildPreprocessor(trainingSet: DataFrame): PipelineModel = {
    val remover = new DiacriticRemover().setInputCol("text").setOutputCol("x")
    val inputTokenizer = new RegexTokenizer().setInputCol("x").setOutputCol("x0").setPattern(config.delimiters).setToLowercase(true)
    val inputConverter = new TokenConverter().setInputCol("x0").setOutputCol("xs")
    val outputTokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("y0").setPattern(config.delimiters).setToLowercase(true)
    val outputConverter = new TokenConverter().setInputCol("y0").setOutputCol("y1")
    val difference = new Difference().setInputCol("y1").setOutputCol("ys")
    val inputVectorizer = new CountVectorizer().setInputCol("xs").setOutputCol("features").setMinDF(config.minFrequency).setBinary(true)
    val outputVectorizer = new CountVectorizer().setInputCol("ys").setOutputCol("labels").setMinDF(config.minFrequency).setBinary(true)
    new Pipeline().setStages(Array(remover, inputTokenizer, inputConverter, outputTokenizer, outputConverter, difference, inputVectorizer, outputVectorizer)).fit(trainingSet)
  }

  /**
    * Creates a main pipeline network model to transform an input vector
    * to an output vector of corresponding sizes.
    *
    * LookupTable => [BiGRU, BiGRU,...] => Linear => LogSoftMax
    *
    * @param inputSizes
    * @param outputSize
    * @return a sequential model.
    */
  override def transducer(inputSizes: Array[Int], outputSize: Int): Module[Float] = {
    val model = Sequential[Float]()
    val lookup = new LookupTable[Float](inputSizes(0) + 1, config.lookupWordSize, paddingValue = 1f, maskZero = true)
    model.add(lookup)
    val merge = JoinTable[Float](2, 2).asInstanceOf[AbstractModule[Table, Tensor[Float], Float]]
    val first = BiRecurrent[Float](merge)
    if (config.gru)
      first.add(GRU(config.lookupWordSize, config.hiddenUnits, config.dropout))
    else first.add(LSTM(config.lookupWordSize, config.hiddenUnits, config.dropout))
    val others = (1 until config.layers).map(_ => {
      val merge = JoinTable[Float](2, 2).asInstanceOf[AbstractModule[Table, Tensor[Float], Float]]
      val other = BiRecurrent[Float](merge)
      if (config.gru)
        other.add(GRU(2*config.hiddenUnits, config.hiddenUnits, config.dropout))
      else other.add(LSTM(2*config.hiddenUnits, config.hiddenUnits, config.dropout))
    })
    model.add(first)
    if (others.nonEmpty)
      others.foreach(model.add(_))
    model.add(TimeDistributed(Linear(2*config.hiddenUnits, outputSize)))
      .add(TimeDistributed(LogSoftMax()))
    logger.info(model.toString())
    model
  }

  /**
    * Converts a list of sequences into a sample to feed into BigDL module.
    * @param df data set
    * @param training training mode or test mode
    * @return a RDD of samples.
    */
  private def toSample(df: DataFrame, training: Boolean = true): RDD[Sample[Float]] = {
    df.select("input", "output").rdd.map { row =>
      val x = row.get(0).asInstanceOf[Seq[Int]].toArray.map(e => e.toFloat + 1)
      if (training) {
        val y = row.get(1).asInstanceOf[Seq[Int]].toArray.map(e => e.toFloat)
        Sample(featureTensor = Tensor(x, Array(x.size)), labelTensor = Tensor(y, Array(y.size)))
      } else Sample(featureTensor = Tensor(x, Array(x.size)))
    }
  }

  /**
    * Trains a BigDL model on a training set.
    * @param trainingSet the training set
    * @param validationSet the development set
    * @return a sequential model.
    */
  override def train(trainingSet: DataFrame, validationSet: DataFrame): Module[Float] = {
    val preprocessor = buildPreprocessor(trainingSet)
    logger.info("Saving the Spark pre-processing pipeline...")
    val modelSt = "M" + config.modelType + (if (config.gru) "G"; else "L") + config.layers + "H" + config.hiddenUnits
    val path = config.modelPath + s"${modelSt}/"
    preprocessor.write.overwrite().save(path)
    val inputLabels = preprocessor.stages(6).asInstanceOf[CountVectorizerModel].vocabulary
    val outputLabels = preprocessor.stages(7).asInstanceOf[CountVectorizerModel].vocabulary
    val numInputLabels = inputLabels.size
    val numOutputLabels = outputLabels.size
    logger.info("numInputLabels = " + numInputLabels)
    logger.info("inputLabels = " + inputLabels.mkString(", "))
    logger.info("numOutputLabels = " + numOutputLabels)
    logger.info("outputLabels = " + outputLabels.mkString(", "))

    val tokenSequencer = new SequenceVectorizer(inputLabels.zipWithIndex.toMap).setInputCol("xs").setOutputCol("input")
    val labelSequencer = new SequenceVectorizer(outputLabels.zipWithIndex.toMap).setInputCol("ys").setOutputCol("output")

    val transformers = Array(preprocessor, tokenSequencer, labelSequencer)
    var (training, validation) = (trainingSet, validationSet)
    transformers.foreach(t => {
      training = t.transform(training)
      validation = t.transform(validation)
    })
    val trainingRDD = toSample(training, true)
    val validationRDD = toSample(validation, true)

    val model = transducer(Array(numInputLabels), numOutputLabels)
    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainingRDD,
      criterion = TimeDistributedCriterion(ClassNLLCriterion[Float](paddingValue = 0), true),
      batchSize = config.batchSize,
      featurePaddingParam = paddingX,
      labelPaddingParam = paddingY
    )

    val trainSummary = TrainSummary(appName = "VDR", logDir = "/tmp/")
    val validationSummary = ValidationSummary(appName = "VDR", logDir = "/tmp/")

    logger.info("Training a RNN transducer model...")
    optimizer.setOptimMethod(new Adagrad[Float](learningRate = config.learningRate, learningRateDecay = 1E-3))
      .setEndWhen(Trigger.maxEpoch(config.epochs))
      .setValidation(Trigger.everyEpoch, validationRDD, Array(new TimeDistributedTop1Accuracy(paddingValue = 0)), config.batchSize,
        featurePaddingParam = paddingX, labelPaddingParam = paddingY)
      .setTrainSummary(trainSummary)
      .setValidationSummary(validationSummary)
      .optimize()
    logger.info("Saving the RNN transducer...")
    model.saveModule(path + "vdg.bigdl", path + "vdg.bin", true)
  }

  override def predict(dataset: DataFrame, preprocessor: PipelineModel, module: Module[Float]): RDD[Row] = {
    val inputLabels = preprocessor.stages(6).asInstanceOf[CountVectorizerModel].vocabulary
    val outputLabels = preprocessor.stages(7).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info("numInputLabels = " + inputLabels.size)
    logger.info("numOutputLabels = " + outputLabels.size)

    val tokenSequencer = new SequenceVectorizer(inputLabels.zipWithIndex.toMap).setInputCol("xs").setOutputCol("input")
    val labelSequencer = new SequenceVectorizer(outputLabels.zipWithIndex.toMap).setInputCol("ys").setOutputCol("output")

    val df = labelSequencer.transform(tokenSequencer.transform(preprocessor.transform(dataset)))
    val rdd = toSample(df, false)

    val predictor = Predictor[Float](module, Some(paddingX))
    val predictions = predictor.predict(rdd).map(activity => {
      val ys = activity.toTensor[Float].split(1).toSeq
      val zs = ys.map(tensor => {
        val k = (0 until tensor.toArray().size).zip(tensor.toArray()).maxBy(p => p._2)._1
        outputLabels(k)
      })
      zs
    })
    df.select("xs", "ys").rdd.zip(predictions).map(p => {
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
    val slices = Array(text)
    val sparkSession = SparkSession.builder().getOrCreate()
    import sparkSession.implicits._
    val input = sparkSession.sparkContext.parallelize(slices).toDF("text")
    val output = test(input, preprocessor, module)
    val same = Set("S", "0")
    output.map(row => {
      val xs = row.getAs[Seq[String]](0) // [no, la, dang, vien, cao, cap]
      val zs = row.getAs[Seq[String]](1) // [ó, à, đả, ê, S, ấ]
      // remove diacritics of zs to match its position in xs
      xs.zip(zs).map(p => {
        if (same.contains(p._2)) p._1 else {
          val s = DiacriticRemover.run(p._2)
          val j = p._1.indexOf(s)
          p._1.substring(0, j) + p._2 + p._1.substring(j + p._2.size)
        }
      })
    }).collect().mkString("\n")
  }
}
