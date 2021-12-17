package vlp.ner

import org.apache.spark.sql.SparkSession

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn.keras.Model

import com.intel.analytics.zoo.pipeline.api.keras.layers._

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.Module

import scopt.OptionParser
import org.slf4j.LoggerFactory
import org.apache.log4j.Level
import org.apache.log4j.Logger

import org.json4s.{DefaultFormats, NoTypeHints}
import org.json4s.jackson.Serialization
import com.intel.analytics.bigdl.nn.{TimeDistributedCriterion, ClassNLLCriterion}
import com.intel.analytics.zoo.pipeline.nnframes.{NNEstimator, NNClassifier, NNModel}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.optim.{Adam, Trigger, Top1Accuracy, ValidationMethod, ValidationResult, AccuracyResult}
import org.apache.spark.ml.linalg.Vectors
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.feature._
import org.apache.spark.ml._
import java.nio.file.{Paths, Files, StandardOpenOption}
import com.intel.analytics.bigdl.dataset.Sample
import org.apache.spark.ml.linalg.DenseVector

/**
  * A neural named entity tagger for Vietnamese.
  * <p/>
  * phuonglh@gmail.com
  * 
  */
class NeuralTagger(sparkSession: SparkSession, config: ConfigNER) extends Serializable {
  val logger = LoggerFactory.getLogger(getClass.getName)
  val prefix = Paths.get(config.modelPath, "gru", config.recurrentSize.toString).toString()

  import sparkSession.implicits._

  def createDataFrame(dataPath: String): DataFrame = {
    val sentences = CorpusReader.readCoNLL(dataPath, config.twoColumns)
    val seqPairs = sentences.map(sentence => {
      val tokens = sentence.tokens
      (tokens.map(_.word).mkString(" "), tokens.map(_.namedEntity).mkString(" "))
    })
    sparkSession.createDataFrame(seqPairs).toDF("x", "y")
  }

  def train(training: DataFrame, test: DataFrame): Module[Float] = {
    // build a preprocessing pipeline and determine the dictionary and vocab size
    val wordTokenizer = new Tokenizer().setInputCol("x").setOutputCol("words")
    val wordCountVectorizer = new CountVectorizer().setInputCol("words").setOutputCol("wordVector").setMinDF(config.minFrequency)
    val labelTokenizer = new Tokenizer().setInputCol("y").setOutputCol("labels")
    val labelCountVectorizer = new CountVectorizer().setInputCol("labels").setOutputCol("labelVector").setMinDF(config.minFrequency)
    val wordShaper = new WordShaper().setInputCol("words").setOutputCol("shapes")
    val shapeIndexer = new CountVectorizer().setInputCol("shapes").setOutputCol("shapeVector")
    val mentionExtractor = new MentionExtractor().setInputCols(Array("words", "labels")).setOutputCol("mentions")
    val mentionCountVectorizer = new CountVectorizer().setInputCol("mentions").setOutputCol("mentionVector")

    val preprocessingPipeline = new Pipeline().setStages(Array(wordTokenizer, wordCountVectorizer, labelTokenizer, labelCountVectorizer, 
      wordShaper, shapeIndexer, mentionExtractor, mentionCountVectorizer))
    val preprocessingPipelineModel = preprocessingPipeline.fit(training)  
    val (trainingAlpha, testAlpha) = (preprocessingPipelineModel.transform(training), preprocessingPipelineModel.transform(test))
    if (config.verbose) {
      testAlpha.show()
    }
    preprocessingPipelineModel.write.overwrite.save(Paths.get(config.modelPath, "gru", config.recurrentSize.toString()).toString())

    val wordDictionary = preprocessingPipelineModel.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex.toMap
    val vocabSize = wordDictionary.size
    val labelDictionary = preprocessingPipelineModel.stages(3).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex.toMap
    val labelSize = labelDictionary.size
    val shapeDictionary = preprocessingPipelineModel.stages(5).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex.toMap
    val shapeSize = shapeDictionary.size
    val mentionDictionary = preprocessingPipelineModel.stages(7).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex.toMap
    val mentionSize = mentionDictionary.size
    logger.info(s"vocabSize = ${vocabSize}")
    logger.info(labelDictionary.toString)
    logger.info(shapeDictionary.toString)
    logger.info(s"mentionSize = ${mentionSize}")

    // transform sequences of words/labels/shapes/mentions into vectors of indices for use in the DL model
    val wordSequenceVectorizer = new SequenceVectorizer(wordDictionary, config.maxSequenceLength).setInputCol("words").setOutputCol("word")
    val labelSequenceVectorizer = new SequenceVectorizer(labelDictionary, config.maxSequenceLength, false, -1).setInputCol("labels").setOutputCol("label")
    val shapeSequenceVectorizer = new SequenceVectorizer(shapeDictionary, config.maxSequenceLength).setInputCol("shapes").setOutputCol("shape")
    val mentionSequenceVectorizer = new SequenceVectorizer(mentionDictionary, config.maxSequenceLength, binary = true).setInputCol("mentions").setOutputCol("mention")
    val vectorAssembler = new VectorAssembler().setInputCols(Array("word", "shape", "mention")).setOutputCol("features")

    val pipeline = new Pipeline().setStages(Array(wordSequenceVectorizer, labelSequenceVectorizer, shapeSequenceVectorizer, mentionSequenceVectorizer, vectorAssembler))
    val pipelineModel = pipeline.fit(trainingAlpha)
    val (trainingBeta, testBeta) = (pipelineModel.transform(trainingAlpha), pipelineModel.transform(testAlpha))
    if (config.verbose) {
      testBeta.show()
    }
    // train a DL model
    val featureSize = 3*config.maxSequenceLength
    val model = buildModel(vocabSize + 1, shapeSize + 1, labelSize, featureSize)
    val trainSummary = TrainSummary(appName = config.recurrentSize.toString, logDir = "/tmp/ner/" + config.language)
    val validationSummary = ValidationSummary(appName = config.recurrentSize.toString, logDir = "/tmp/ner/" + config.language)
    val classifier = NNEstimator(model, TimeDistributedCriterion(ClassNLLCriterion[Float](paddingValue = -1)), Array(featureSize), Array(config.maxSequenceLength))
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setBatchSize(config.batchSize)
      .setOptimMethod(new Adam(0.001))
      .setMaxEpoch(config.epochs)
      .setTrainSummary(trainSummary)
      .setValidationSummary(validationSummary)
      .setValidation(Trigger.everyEpoch, testBeta, Array(new TimeDistributedTop1Accuracy()), config.batchSize)
    classifier.fit(trainingBeta)
    model.saveModule(prefix + ".bigdl", prefix + ".bin", true)
  }

  /**
    * Constructs a sequential model for NER using Keras-style layers. We use both word embeddings,  
    * shape embeddings and binary-valued mentions in a pipeline as follows:
    * [w1,...wN, s1,...,sN, b1,...,bN] => Reshape(3, N) => (Embedding, Embedding, Select) => Merge('concat') => GRU/BiGRU => Dense('softmax')
    *
    * @param vocabSize
    * @param shapeSize
    * @param labelSize
    * @param featureSize
    * @return a BigDL Keras-style model
    */
  def buildModel(vocabSize: Int, shapeSize: Int, labelSize: Int, featureSize: Int): Module[Float] = {
    val inputNode = Input(inputShape = Shape(featureSize))
    val reshapeNode = Reshape(Array(3, featureSize/3)).inputs(inputNode)
    
    val wordSelectNode = Select(1, 0).inputs(reshapeNode)
    val wordEmbeddingNode = Embedding(vocabSize, config.wordEmbeddingSize).inputs(wordSelectNode)
    
    val shapeSelectNode = Select(1, 1).inputs(reshapeNode)
    val shapeEmbeddingNode = Embedding(shapeSize, config.shapeEmbeddingSize).inputs(shapeSelectNode)

    val mentionSelectNode = Select(1, 2).inputs(reshapeNode)
    val addOneNode = AddConstant(1).inputs(mentionSelectNode)
    val mentionEmbeddingNode = Embedding(3, 2).inputs(addOneNode)

    val mergeNode = Merge(mode = "concat").inputs(Array(wordEmbeddingNode, shapeEmbeddingNode, mentionEmbeddingNode))
    
    val recurrentNode = if (!config.bidirectional) {
      GRU(config.recurrentSize, returnSequences = true).inputs(mergeNode)
    } else {
      Bidirectional(GRU(config.recurrentSize, returnSequences = true), mergeMode = "concat").inputs(mergeNode)
    }
    val outputNode = if (config.outputSize > 0) {
      val denseNode = Dense(config.outputSize, activation = "relu").inputs(recurrentNode)
      Dense(labelSize, activation = "softmax").inputs(denseNode)
    } else Dense(labelSize, activation = "softmax").inputs(recurrentNode)
    val model = Model(inputNode, outputNode)
    model
  }

  /**
   * Predicts label sequence given word sequence. The input data frame has 'x' column.
  */
  def predict(input: DataFrame, preprocessor: PipelineModel, module: Module[Float]): (DataFrame, DataFrame) = {
    val alpha = preprocessor.transform(input)

    val wordDictionary = preprocessor.stages(1).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex.toMap
    val wordSequenceVectorizer = new SequenceVectorizer(wordDictionary, config.maxSequenceLength).setInputCol("words").setOutputCol("word")
    val shapeDictionary = preprocessor.stages(5).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex.toMap
    val shapeSequenceVectorizer = new SequenceVectorizer(shapeDictionary, config.maxSequenceLength).setInputCol("shapes").setOutputCol("shape")
    val mentionDictionary = preprocessor.stages(7).asInstanceOf[CountVectorizerModel].vocabulary.zipWithIndex.toMap
    val mentionSequenceVectorizer = new SequenceVectorizer(mentionDictionary, config.maxSequenceLength, binary = true).setInputCol("mentions").setOutputCol("mention")
    val vectorAssembler = new VectorAssembler().setInputCols(Array("word", "shape", "mention")).setOutputCol("features")
    val pipeline = new Pipeline().setStages(Array(wordSequenceVectorizer, shapeSequenceVectorizer, mentionSequenceVectorizer, vectorAssembler))
    val labels = preprocessor.stages(3).asInstanceOf[CountVectorizerModel].vocabulary
    val labelMap = (0 until labels.size).zip(labels).toMap

    val beta = pipeline.fit(alpha).transform(alpha)
    val rdd = beta.select("features").rdd.map{ row => 
      val x = row.get(0).asInstanceOf[DenseVector]
      Sample(featureTensor = Tensor(x.values.map(_.toFloat), Array(3*config.maxSequenceLength)))
    }
    val predictions = module.predict(rdd).map(activity => {
      // split a tensor of shape (maxSequenceLength x numLabels) into a sequence of 1d tensors, each of length numLabels.
      val ys = activity.toTensor[Float].split(1).toSeq
      // find argmax (positions of maximum values)
      val zs = ys.map(tensor => (0 until tensor.toArray().size).zip(tensor.toArray()).maxBy(p => p._2)._1)
      zs.map(labelMap(_))
    })
    if (config.verbose) {
      alpha.show()
      beta.select("word", "shape", "mention", "features").show()
    }
    (beta.select("words", "y"), predictions.toDF("z"))
  }

  def predict(inputPathCoNLL: String, preprocessor: PipelineModel, module: Module[Float], outputPath: String): Unit = {
    val inputDF = createDataFrame(inputPathCoNLL)
    val (df1, df2) = predict(inputDF, preprocessor, module)
    val ys = df1.select("words", "y").map(row => {
      val n = row.getAs[Seq[String]](0).size
      row.getAs[String](1).split(" ").take(n)
    }).collect()
    val zs = df2.map(row => row.getAs[Seq[String]](0)).collect()
    val ss = ys.zip(zs).map{case (y, z) => 
      val s = y.zip(z).map { case (a, b) => a + " " + b.toUpperCase() }.mkString("\n") 
      s + "\n"
    }
    import scala.collection.JavaConversions._
    Files.write(Paths.get(outputPath), ss.toList, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }
}

/**
  * Time-distributed top 1 accuracy.
  *
  * @param paddingValue padding value for label sequence
  * @param ev
  */
class TimeDistributedTop1Accuracy(paddingValue: Int = -1)(implicit ev: TensorNumeric[Float]) extends ValidationMethod[Float] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    var correct = 0
    var count = 0
    val _output = output.asInstanceOf[Tensor[Float]]
    val _target = target.asInstanceOf[Tensor[Float]]
    _output.split(1).zip(_target.split(1)).foreach { case (tensor, ys) => 
      val zs = tensor.split(1).map { t =>
        val values = t.toArray()
        val k = (0 until values.size).zip(values).maxBy(p => p._2)._1
        k + 1 // one-based label index
      }
      // filter the padded value in the gold target before comparing with the prediction
      val c = ys.toArray().filter(e => e != paddingValue).zip(zs)
        .map(p => if (p._1 == p._2) 1 else 0)
      correct += c.sum
      count += c.size
    }
    new AccuracyResult(correct, count)
  }
  override def format(): String = "TimeDistributedTop1Accuracy"
}

object NeuralTagger {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    implicit val formats = Serialization.formats(NoTypeHints)

    val parser = new OptionParser[ConfigNER]("vlp.ner.NeuralTagger") {
      head("vlp.ner.NeuralTagger", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/predict")
      opt[String]('Z', "executorMemory").action((x, conf) => conf.copy(executorMemory = x)).text("executor memory, default is 8g")
      opt[String]('l', "language").action((x, conf) => conf.copy(language = x)).text("language, either 'vie' or 'eng'")
      opt[Int]('b', "batchSize").action((x, conf) => conf.copy(batchSize = x)).text("batch size")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('u', "numFeatures").action((x, conf) => conf.copy(numFeatures = x)).text("number of features")
      opt[String]('d', "trainingPath").action((x, conf) => conf.copy(trainingPath = x)).text("training path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/zoo/tcl/'")
      opt[Int]('w', "wordEmbeddingSize").action((x, conf) => conf.copy(wordEmbeddingSize = x)).text("word embedding size, default is 100")
      opt[Int]('s', "shapeEmbeddingSize").action((x, conf) => conf.copy(shapeEmbeddingSize = x)).text("shape embedding size, default is 10")
      opt[Int]('r', "recurrentSize").action((x, conf) => conf.copy(recurrentSize = x)).text("output size of the recurrent layer")
      opt[Int]('o', "outputSize").action((x, conf) => conf.copy(outputSize = x)).text("output size of the dense layer")
      opt[Int]('n', "maxSequenceLength").action((x, conf) => conf.copy(maxSequenceLength = x)).text("maximum sequence length of a sentence")
      opt[Int]('k', "epochs").action((x, conf) => conf.copy(epochs = x)).text("number of epochs")
      opt[Unit]('v', "verbose").action((x, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[Unit]('q', "bidirectional").action((x, conf) => conf.copy(bidirectional = true)).text("bidirectional mode")
      opt[Unit]('j', "twoColumns").action((x, conf) => conf.copy(twoColumns = true)).text("two-column mode")
    }
    parser.parse(args, ConfigNER()) match {
      case Some(config) =>
        val sparkConfig = Engine.createSparkConf()
          .setMaster(config.master)
          .set("spark.executor.memory", config.executorMemory)
          .set("spark.executor.extraJavaOptions", "-Dbigdl.engineType=mkldnn -Dcom.github.fommil.netlib.BLAS=com.intel.mkl.MKLBLAS -Dcom.github.fommil.netlib.LAPACK=com.intel.mkl.MKLLAPACK")
          .set("spark.driver.extraJavaOptions", "-Dbigdl.engineType=mkldnn")
          .setAppName("ner.NeuralTagger")
        val sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate()
        val sparkContext = sparkSession.sparkContext
        Engine.init

        println(Serialization.writePretty(config))

        val tagger = new NeuralTagger(sparkSession, config)
        val (training, test) = (tagger.createDataFrame(config.trainingPath), tagger.createDataFrame(config.validationPath))
        training.show()
        println(training.count())
        test.show()
        println(test.count())
        config.mode match {
          case "train" => 
            val module = tagger.train(training, test)
            val preprocessor = PipelineModel.load(Paths.get(config.modelPath, "gru", config.recurrentSize.toString).toString())
            tagger.predict(config.trainingPath, preprocessor, module, config.modelPath + "/" + s"train-${config.recurrentSize}-${config.bidirectional}.gru")
            tagger.predict(config.validationPath, preprocessor, module, config.modelPath + "/" + s"dev-${config.recurrentSize}-${config.bidirectional}.gru")
          case "predict" => 
            val preprocessor = PipelineModel.load(Paths.get(config.modelPath, "gru", config.recurrentSize.toString()).toString())
            val module = com.intel.analytics.bigdl.nn.Module.loadModule[Float](tagger.prefix + ".bigdl", tagger.prefix + ".bin")
            tagger.predict(config.trainingPath, preprocessor, module, config.modelPath + "/" + s"train-${config.recurrentSize}-${config.bidirectional}.gru")
            tagger.predict(config.validationPath, preprocessor, module, config.modelPath + "/" + s"dev-${config.recurrentSize}-${config.bidirectional}.gru")
          case "extract" =>
            val df = training.union(test)
            val wordTokenizer = new Tokenizer().setInputCol("x").setOutputCol("words")
            val labelTokenizer = new Tokenizer().setInputCol("y").setOutputCol("labels")
            val mentionExtractor = new MentionExtractor().setInputCols(Array("words", "labels")).setOutputCol("mentions")
            val pipeline = new Pipeline().setStages(Array(wordTokenizer, labelTokenizer, mentionExtractor))
            val output = pipeline.fit(df).transform(df).select("mentions")
            import sparkSession.implicits._
            val count = output.flatMap(row => row.getAs[Seq[String]](0)).groupBy("value").count().sort($"count".desc)
            count.show(50, false)
        }
        sparkSession.stop()
      case None => 
    }
  }
}
