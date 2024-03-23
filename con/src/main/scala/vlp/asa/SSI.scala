package vlp.asa

import java.nio.file.{Files, Paths, StandardOpenOption}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.feature.{RegexTokenizer, CountVectorizer, StringIndexer, IDF, CountVectorizerModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.functions._
import com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach
import com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings
import com.johnsnowlabs.nlp.embeddings.{DeBertaEmbeddings, DistilBertEmbeddings, XlnetEmbeddings}
import com.johnsnowlabs.nlp.embeddings.{BertSentenceEmbeddings, RoBertaSentenceEmbeddings, XlmRoBertaSentenceEmbeddings, UniversalSentenceEncoder}
import com.johnsnowlabs.nlp.Annotation


case class Entity(text: String, start: Long, end: Long, labels: Array[String], reference: String)
case class AspectSentiment(aspect: String, sentiment: String, reference: String)
case class Element(
  id: String,
  `type`: String,
  text: String,
  tags: Array[String],
  entities: Array[Entity],
  aspect_sentiments: Array[AspectSentiment],
)
case class Sample(
  id: String,
  `type`: String,
  text: String,
  tags: Array[String],
  entities: Array[Entity],
  aspect_sentiments: Array[AspectSentiment],
  comments: Array[Element]
)

case class Instance(postText: String, postSentiments: Array[String], commentText: String, commentSentiments: Array[String])

case class ConfigSSI(
  modelType: String = "d",
  vocabSize: Int = 1300,
  learningRate: Float = 5E-4f,
  hiddenSize: Int = 256,
  batchSize: Int = 32,
  maxIter: Int = 30,
  delimiters: String = """([\s,'")(;!”“*>\]\[=•]+|[\.,:?]+\s+|([\-_]{2,}))""",
  minDF: Int = 2,
  modelPath: String = "bin/ssi/",
  validPath: String = "dat/ssi/val/"
)

object SSI {

  /**
    * Reads in the data set and performs preprocessing as follows: (1) Each post or comment is assigned 
    * the sentiment which is the most frequent (say, if a comment which has 3 positive labels and 2 negative labels then it
    * is consider positive); if there is not any sentiments, the NA label is returned. (2) Each post is flat-mapped with its comments.
    *
    * @param spark
    * @param path
    */
  def readData(spark: SparkSession, path: String): DataFrame = {
    import spark.implicits._
    val df = spark.read.option("multiline", "true").json(path).as[Sample]
    // find the most frequent sentiment of a document, or NA if the document does not have any sentiment.
    def f(ass: Array[AspectSentiment]) = {
      val sentiments = ass.map(e => e.sentiment).groupBy(identity).toList.map(p => (p._1, p._2.size)).sortBy(-_._2)
      if (sentiments.isEmpty) "NA" else sentiments.head._1
    }
    // flat map the dataset based on comments
    df.flatMap(sample => sample.comments.map(c => (sample.text, f(sample.aspect_sentiments), c.text, f(c.aspect_sentiments)))).toDF("postText", "postSentiment", "commentText", "commentSentiment")
  }

  def filterComments(df: DataFrame): DataFrame = {
    df.filter(col("commentSentiment") =!= "NA").filter(col("commentSentiment") =!= "unrelated")
  }

  def fitMLP(df: DataFrame, config: ConfigSSI) = {
    val labelIndexer = new StringIndexer().setInputCol("commentSentiment").setOutputCol("label")
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern(config.delimiters)
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("counts").setVocabSize(config.vocabSize).setMinDF(config.minDF)
    val idf = new IDF().setInputCol("counts").setOutputCol("features")
    val classifier = new MultilayerPerceptronClassifier().setLayers(Array(config.vocabSize, config.hiddenSize, 3)).setBlockSize(config.batchSize).setMaxIter(config.maxIter).setSeed(1234L)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, vectorizer, idf, classifier))
    pipeline.fit(df)
  }

  def fitJSL(trainingDF: DataFrame, developmentDF: DataFrame, config: ConfigSSI): PipelineModel = {
    // use a vectorizer to get label vocab
    val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
    val embeddings = config.modelType match {
      case "b" => BertSentenceEmbeddings.pretrained("sent_bert_multi_cased", "xx").setInputCols("document").setOutputCol("embeddings")
      case "u" => UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx").setInputCols("document").setOutputCol("embeddings")
      case "r" => RoBertaSentenceEmbeddings.pretrained().setInputCols("document").setOutputCol("embeddings") // this is for English only
      case "x" => XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_base", "xx").setInputCols("document").setOutputCol("embeddings")
      case "d" => DeBertaEmbeddings.pretrained("deberta_embeddings_vie_small", "vie").setInputCols("document", "token").setOutputCol("xs")
      case "s" => DistilBertEmbeddings.pretrained("distilbert_base_cased", "vi").setInputCols("document", "token").setOutputCol("xs")
      case _ => UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx").setInputCols("document").setOutputCol("embeddings")
    }
    var stages = Array(document, tokenizer, embeddings)
    val sentenceEmbedding = new SentenceEmbeddings().setInputCols("document", "xs").setOutputCol("embeddings")
    if (Set("d", "s").contains(config.modelType)) 
      stages = stages ++ Array(sentenceEmbedding)
    val classifier = new ClassifierDLApproach().setInputCols("embeddings").setOutputCol("category").setLabelColumn("commentSentiment").setBatchSize(config.batchSize).setMaxEpochs(config.maxIter).setLr(config.learningRate)
    // train a preprocessor 
    val preprocessor = new Pipeline().setStages(stages)
    val preprocessorModel = preprocessor.fit(trainingDF)
    // use the preprocessor pipeline to transform the development set 
    val df = preprocessorModel.transform(developmentDF)
    df.write.mode("overwrite").parquet(config.validPath)
    classifier.setTestDataset(config.validPath)
    // train the whole pipeline and return a model
    stages = stages ++ Array(classifier)
    val pipeline = new Pipeline().setStages(stages)
    val model = pipeline.fit(trainingDF)
    val validPrediction = model.transform(developmentDF)
    validPrediction.select("commentSentiment", "category.result").show(false)
    model
  }

  /**
    * Evaluates the model on a data frame. The default metric is "weightedFMeasure" (beta=1.0)
    *
    * @param model
    * @param df
    * @param metricName
    * @return a double
    */
  def evaluate(model: PipelineModel, df: DataFrame, metricName: String="f1"): Double = {
    val result = model.transform(df)
    result.show
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName(metricName)
    evaluator.evaluate(predictionAndLabels)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(getClass().getName()).setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    val path = "dat/ssi/project-28-at-2023-02-21-10-28-518a1b2d.json-formatted-4.json"
    val af = readData(spark, path)
    val bf = filterComments(af)
    bf.show
    // concatenate two text columns
    val df = bf.withColumn("text", concat(col("postText"), col("commentText")))
    val Array(trainDF, validDF) = df.randomSplit(Array(0.8, 0.2), 220712L)
    val opt = args(0)
    if (opt == "mlp") { // Spark MLP
      val model = fitMLP(trainDF, ConfigSSI())
      val trainF1 = evaluate(model, trainDF)
      val validF1 = evaluate(model, validDF)
      println(s"trainF1 = $trainF1, validF1 = $validF1")
      val vocab = model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
      println("\t Top 50 tokens: " + vocab.take(50).mkString(" "))
      println("\tLast 50 tokens: " + vocab.takeRight(50).mkString(" "))
      println(s"vocabSize = ${vocab.size}")
    } else if (opt == "jsl") { // JohnSnowLab
      val model = fitJSL(trainDF, validDF, ConfigSSI())
    } else {
      println("Specify a method to run: mlp/jsl")
    }
    // print some stats about training data
    val countDF = trainDF.groupBy("commentSentiment").count
    countDF.show
    spark.stop()
  }
}
