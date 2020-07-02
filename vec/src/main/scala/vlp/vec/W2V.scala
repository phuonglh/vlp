package vlp.vec

import vlp.tok.TokenizerTransformer
import org.apache.log4j._
import org.apache.spark.ml.feature.{Tokenizer, Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import org.slf4j.LoggerFactory
import scopt.OptionParser

/**
  * phuonglh, April, 2018; updated December 2019.
  *
  */
class W2V(spark: SparkSession, config: ConfigW2V) extends Serializable {
  final val logger = LoggerFactory.getLogger(getClass.getName)

  def train: Unit = {
    import spark.implicits._
    val dataset = spark.sparkContext.textFile(config.data).filter(_.trim.size > config.minLength).toDF("text")
    dataset.cache()
    logger.info("#(sentences) = " + dataset.count())
    val vietnameseTokenizer = new TokenizerTransformer().setInputCol("text").setOutputCol("tokenized").setConvertPunctuation(true)
    val tokenizer = new Tokenizer().setInputCol("tokenized").setOutputCol("tokens")
    val w2v = new Word2Vec().setInputCol("tokens").setOutputCol("vector")
      .setMinCount(config.minFrequency).setVectorSize(config.dimension).setWindowSize(config.windowSize)
      .setMaxIter(config.iterations).setSeed(220712)
    val pipeline = new Pipeline().setStages(Array(vietnameseTokenizer, tokenizer, w2v))
    logger.info("Training the model. Please wait...")
    val model = pipeline.fit(dataset)
    model.write.overwrite().save(config.modelPath)
    if (config.verbose) {
      val output = model.transform(dataset)
      output.show(false)
    }
    // get the trained word to vec model
    val m = model.stages(2).asInstanceOf[Word2VecModel]
    val pw = new java.io.PrintWriter(new java.io.File(config.output))
    try {
      m.getVectors.collect().foreach { row =>
        val line = row.getAs[String](0) + "\t" + row.getAs[DenseVector](1)
        pw.write(line)
        pw.write("\n")
      }
    } finally {
      pw.close()
    }
  }

  def eval: Unit = {
    val model = PipelineModel.load(config.modelPath)
    val w2v = model.stages(2).asInstanceOf[Word2VecModel]
    val words = Array("đồng_nai", "hà_nội", "giàu", "đầu_tư", "thành_phố", "văn_bản")
    words.foreach { word =>
      val synonyms = w2v.findSynonyms(word, 10)
      logger.info("synonyms of '" + word + "':\n")
      synonyms.show(false)
    }
  }
}

object W2V {
  final val logger = LoggerFactory.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val parser = new OptionParser[ConfigW2V]("vlp.vec") {
      head("vlp.vec", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('l', "minLength").action((x, conf) => conf.copy(minLength = x)).text("min sentence length in characters, default is 20")
      opt[Int]('w', "windowSize").action((x, conf) => conf.copy(windowSize = x)).text("windows size, default is 5")
      opt[String]('d', "data").action((x, conf) => conf.copy(data = x)).text("data path")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is '/dat/vlp/vec/'")
      opt[String]('o', "output").action((x, conf) => conf.copy(output = x)).text("output path")
    }
    parser.parse(args, ConfigW2V()) match {
      case Some(config) =>
        val sparkSession = SparkSession.builder().appName(getClass.getName).master(config.master).config("spark.executor.memory", "8g").getOrCreate()
        implicit val formats = Serialization.formats(NoTypeHints)
        logger.info(Serialization.writePretty(config))
        val w2v = new W2V(sparkSession, config)
        config.mode match {
          case "train" => {
            w2v.train
          }
          case "eval" => {
            w2v.eval
          }
        }
        sparkSession.stop()
      case None => {}
    }
  }
}
