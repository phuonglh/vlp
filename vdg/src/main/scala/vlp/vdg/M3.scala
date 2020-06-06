package vlp.vdg

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{PaddingParam, Sample}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.optim.{Adagrad, Optimizer, Predictor, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

/**
  * The third model of VDG which uses both token input and char input.
  *
  * phuonglh@gmail.com
  *
  * @param config
  */
class M3(config: ConfigVDG) extends M2(config){
  override val paddingX = PaddingParam[Float](Some(Array(Tensor(T(1f)), Tensor(T(1f)))))

  override def buildPreprocessor(trainingSet: DataFrame): PipelineModel = {
    val remover = new DiacriticRemover().setInputCol("text").setOutputCol("x")
    val inputWordTokenizer = new RegexTokenizer().setInputCol("x").setOutputCol("x0").setPattern(config.delimiters).setToLowercase(true)
    val tokenConverter = new TokenConverter().setInputCol("x0").setOutputCol("xt")
    val stringMaker = new StringMaker().setInputCol("xt").setOutputCol("s")
    val inputCharTokenizer = new RegexTokenizer().setInputCol("s").setOutputCol("xc").setPattern(".").setGaps(false).setToLowercase(true)

    val outputTokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("y0").setPattern(config.delimiters).setToLowercase(true)
    val outputConverter = new TokenConverter().setInputCol("y0").setOutputCol("y1")
    val difference = new Difference().setInputCol("y1").setOutputCol("ys")
    val inputWordVectorizer = new CountVectorizer().setInputCol("xt").setOutputCol("tokens").setMinDF(config.minFrequency).setBinary(true)
    val inputCharVectorizer = new CountVectorizer().setInputCol("xc").setOutputCol("chars").setMinDF(config.minFrequency).setBinary(true)
    val outputVectorizer = new CountVectorizer().setInputCol("ys").setOutputCol("labels").setMinDF(config.minFrequency).setBinary(true)
    new Pipeline().setStages(Array(remover, inputWordTokenizer, tokenConverter, stringMaker,
      inputCharTokenizer, outputTokenizer, outputConverter, difference, inputWordVectorizer, inputCharVectorizer, outputVectorizer)).fit(trainingSet)
  }

  /**
    * Builds the core DL model, which is a transducer.
    *
    * @param inputSizes
    * @param outputSize
    * @return a sequential model.
    */
  override def transducer(inputSizes: Array[Int], outputSize: Int): Module[Float] = {
    val parallel = ParallelTable()
    val ws = Sequential[Float]()
      .add(DenseToSparse(propagateBack = false))
      .add(LookupTableSparse[Float](inputSizes(0) + 1, config.lookupWordSize, "mean"))
    val wordModule = Sequential[Float]()
      .add(SplitTable(1, 3))
      .add(new MapTableNoAcc(ws))
      .add(Pack(1))
    val cs = Sequential[Float]()
      .add(DenseToSparse(propagateBack = false))
      .add(LookupTableSparse[Float](inputSizes(1) + 1, config.lookupCharacterSize, "mean"))
    val charModule = Sequential[Float]()
      .add(SplitTable(1, 3))
      .add(new MapTableNoAcc(cs))
      .add(Pack(1))

    val parallelNode = parallel.add(wordModule).add(charModule).setName("parallel").inputs()
    val joinNode = JoinTable(2,2).inputs(parallelNode)

    val merge = JoinTable[Float](2, 2).asInstanceOf[AbstractModule[Table, Tensor[Float], Float]]
    val recurrent = BiRecurrent[Float](merge)
    // first recurrent layer
    val first = BiRecurrent[Float](merge)
    if (config.gru)
      first.add(GRU(config.lookupWordSize + config.lookupCharacterSize, config.hiddenUnits, config.dropout))
    else first.add(LSTM(config.lookupWordSize, config.hiddenUnits, config.dropout))
    // other recurrent layers
    val others = (1 until config.layers).map(_ => {
      val merge = JoinTable[Float](2, 2).asInstanceOf[AbstractModule[Table, Tensor[Float], Float]]
      val other = BiRecurrent[Float](merge)
      if (config.gru)
        other.add(GRU(2*config.hiddenUnits, config.hiddenUnits, config.dropout))
      else other.add(LSTM(2*config.hiddenUnits, config.hiddenUnits, config.dropout))
    })
    if (others.nonEmpty)
      others.foreach(recurrent.add(_))
    val recurrentNode = recurrent.add(first).inputs(joinNode)
    val linearNode = TimeDistributed(Linear(2*config.hiddenUnits, outputSize)).inputs(recurrentNode)
    val softmaxNode = TimeDistributed(LogSoftMax()).inputs(linearNode)
    val model = Graph(parallelNode, softmaxNode)
    model.stopGradient(Array("parallel"))
    model
  }

  /**
    * Converts a list of sequences into a sample to feed into BigDL module.
    * @param df a data set
    * @param preprocessor pre-processing pipeline
    * @param training training mode or test mode
    * @return a RDD of samples.
    */
  private def toSample(df: DataFrame, preprocessor: PipelineModel, training: Boolean = true): RDD[Sample[Float]] = {
    val dc = preprocessor.stages(9).asInstanceOf[CountVectorizerModel].vocabulary.size
    df.select("wordInput", "charInput", "output").rdd.map { row =>
      val wordInput = row.get(0).asInstanceOf[Seq[Int]]
      val n = wordInput.size
      // 2-d word tensor of size nx1
      val wordTensor = Tensor(n, 1).fill(0f)
      for (i <- 0 until wordInput.size)
        wordTensor.setValue(i + 1, 1, wordInput(i) + 1)
      // 2-d char tensor
      val charInput = row.get(1).asInstanceOf[Seq[Seq[Int]]]
      val charTensor = Tensor(n, dc).fill(0f)
      try {
        for (i <- 0 until n)
          for (j <- 0 until charInput(i).size)
            charTensor.setValue(i + 1, j + 1, charInput(i)(j) + 1)
        if (training) {
          val output = row.get(2).asInstanceOf[Seq[Int]].toArray.map(e => e.toFloat)
          val tagTensor = Tensor(output, Array(output.size))
          Sample(featureTensors = Array(wordTensor, charTensor), labelTensor = tagTensor)
        } else Sample(featureTensors = Array(wordTensor, charTensor))
      } catch {
        case e: Exception =>
          logger.info("Exception, create a sample with all ones label tensor.")
          logger.info(s"dc = $dc")
          logger.info(s"n = $n")
          logger.info("charInput = " + charInput.mkString(", "))
          println(e.getMessage)
          if (training) {
            val charTensor = Tensor(n, dc).fill(1f)
            val output = row.get(2).asInstanceOf[Seq[Int]].toArray.map(e => e.toFloat)
            val tagTensor = Tensor(output, Array(output.size))
            Sample(featureTensors = Array(wordTensor, charTensor), labelTensor = tagTensor)
          } else Sample(featureTensors = Array(wordTensor, charTensor))
      }
    }
  }

  /**
    * Trains the transducer on a pair of training set and validation set.
    *
    * @param trainingSet
    * @param validationSet
    * @return a module
    */
  override def train(trainingSet: DataFrame, validationSet: DataFrame): Module[Float] = {
    val preprocessor = buildPreprocessor(trainingSet)
    logger.info("Saving the Spark pre-processing pipeline...")
    val modelSt = "M" + config.modelType + (if (config.gru) "G"; else "L") + config.layers + "H" + config.hiddenUnits
    val path = config.modelPath + modelSt + "/"
    preprocessor.write.overwrite().save(path)
    val inputWordLabels = preprocessor.stages(8).asInstanceOf[CountVectorizerModel].vocabulary
    val inputCharLabels = preprocessor.stages(9).asInstanceOf[CountVectorizerModel].vocabulary
    val outputLabels = preprocessor.stages(10).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info(s"#(inputWordLabels) = ${inputWordLabels.size}")
    logger.info(s"#(inputCharLabels) = ${inputCharLabels.size}: ${inputCharLabels.mkString}")
    logger.info(s"#(outputLabels) = ${outputLabels.size}: ${outputLabels.mkString(", ")}")

    val tokenSequencer = new SequenceVectorizer(inputWordLabels.zipWithIndex.toMap).setInputCol("xt").setOutputCol("wordInput")
    val charSequencer = new CharSequencer(inputCharLabels.zipWithIndex.toMap).setInputCol("xt").setOutputCol("charInput")
    val labelSequencer = new SequenceVectorizer(outputLabels.zipWithIndex.toMap).setInputCol("ys").setOutputCol("output")
    val transformers = Array(preprocessor, tokenSequencer, charSequencer, labelSequencer)
    var (training, validation) = (trainingSet, validationSet)
    transformers.foreach(t => {
      training = t.transform(training)
      validation = t.transform(validation)
    })
    training.show(3, false)
    val trainingRDD = toSample(training, preprocessor, true)
    val validationRDD = toSample(validation, preprocessor, true)

    val model = transducer(Array(inputWordLabels.size, inputCharLabels.size), outputLabels.size)
    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainingRDD,
      criterion = TimeDistributedCriterion(ClassNLLCriterion[Float](paddingValue = 0), true),
      batchSize = config.batchSize,
      featurePaddingParam = paddingX,
      labelPaddingParam = paddingY
    )

    val trainSummary = TrainSummary(appName = modelSt, logDir = "/tmp/")
    val validationSummary = ValidationSummary(appName = modelSt, logDir = "/tmp/")

    logger.info("Training a RNN transducer model...")
    optimizer.setOptimMethod(new Adagrad[Float](learningRate = config.learningRate, learningRateDecay = 1E-3))
      .setEndWhen(Trigger.maxEpoch(config.epochs))
      .setValidation(Trigger.everyEpoch, validationRDD, Array(new TimeDistributedTop1Accuracy(paddingValue = 0)),
        config.batchSize, featurePaddingParam = paddingX, labelPaddingParam = paddingY)
      .setTrainSummary(trainSummary)
      .setValidationSummary(validationSummary)
      .optimize()
    logger.info("Saving the RNN transducer...")
    model.saveModule(path + "vdg.bigdl", path + "vdg.bin", true)
    model
  }

  /**
    * Predicts a new data set.
    *
    * @param dataset
    * @param preprocessor
    * @param module
    * @return a RDD of Row objects.
    */
  override def predict(dataset: DataFrame, preprocessor: PipelineModel, module: Module[Float]): RDD[Row] = {
    val inputWordLabels = preprocessor.stages(8).asInstanceOf[CountVectorizerModel].vocabulary
    val inputCharLabels = preprocessor.stages(9).asInstanceOf[CountVectorizerModel].vocabulary
    val outputLabels = preprocessor.stages(10).asInstanceOf[CountVectorizerModel].vocabulary
    logger.info(s"#(inputWordLabels) = ${inputWordLabels.size}")
    logger.info(s"#(inputCharLabels) = ${inputCharLabels.size}: ${inputCharLabels.mkString}")
    logger.info(s"#(outputLabels) = ${outputLabels.size}: ${outputLabels.mkString(", ")}")

    val tokenSequencer = new SequenceVectorizer(inputWordLabels.zipWithIndex.toMap).setInputCol("xt").setOutputCol("wordInput")
    val charSequencer = new CharSequencer(inputCharLabels.zipWithIndex.toMap).setInputCol("xt").setOutputCol("charInput")
    val labelSequencer = new SequenceVectorizer(outputLabels.zipWithIndex.toMap).setInputCol("ys").setOutputCol("output")
    val transformers = Array(preprocessor, tokenSequencer, charSequencer, labelSequencer)
    var df = dataset
    transformers.foreach(t => df = t.transform(df))
    val rdd = toSample(df, preprocessor, false)

    val predictor = Predictor[Float](module, Some(paddingX))
    val predictions = predictor.predict(rdd).map(activity => {
      val ys = activity.toTensor[Float].split(1).toSeq
      val zs = ys.map(tensor => {
        val k = (0 until tensor.toArray().size).zip(tensor.toArray()).maxBy(p => p._2)._1
        outputLabels(k)
      })
      zs
    })
    df.select("xt", "ys").rdd.zip(predictions).map(p => {
      val xs = p._1.getAs[Seq[String]](0)
      val ys = p._1.getAs[Seq[String]](1)
      val zs = p._2
      val n = Math.min(xs.size, config.maxSequenceLength)
      Row(xs.slice(0, n), ys.slice(0, n), zs.slice(0, n))
    })

  }
}
