package vlp.asa

import java.nio.file.{Files, Paths, StandardOpenOption}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._



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
  vocabSize: Int = 1300,
  hiddenSize: Int = 256,
  batchSize: Int = 32,
  maxIter: Int = 30,
  delimiters: String = """([\s,'")(;!”“*>\]\[=•]+|[\.,:?]+\s+|([\-_]{2,}))""",
  minDF: Int = 2,
  modelPath: String = "bin/ssi/"
)

object SSI {

  /**
    * Reads in the data set and performs preprocessing as follows: (1) Each post or comment is assigned 
    * the sentiment which is the most frequent (say, if a comment which has 3 postive labels and 2 negative labels then it 
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

  def fit(df: DataFrame, config: ConfigSSI) = {
    val labelIndexer = new StringIndexer().setInputCol("commentSentiment").setOutputCol("label")
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokens").setPattern(config.delimiters)
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("counts").setVocabSize(config.vocabSize).setMinDF(config.minDF)
    val idf = new IDF().setInputCol("counts").setOutputCol("features")
    val classifier = new MultilayerPerceptronClassifier().setLayers(Array(config.vocabSize, config.hiddenSize, 3)).setBlockSize(config.batchSize).setMaxIter(config.maxIter).setSeed(1234L)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, vectorizer, idf, classifier))
    pipeline.fit(df)
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
    val model = fit(trainDF, ConfigSSI())
    val trainF1 = evaluate(model, trainDF)
    val validF1 = evaluate(model, validDF)
    println(s"trainF1 = $trainF1, validF1 = $validF1")
    val vocab = model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    println("\t Top 50 tokens: " + vocab.take(50).mkString(" "))
    println("\tLast 50 tokens: " + vocab.takeRight(50).mkString(" "))
    println(s"vocabSize = ${vocab.size}")
    spark.stop()
  }
}
