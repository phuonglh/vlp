package vlp.vec

import org.apache.log4j._
import org.apache.spark.ml.feature.{Tokenizer, Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import org.slf4j.LoggerFactory
import scopt.OptionParser
import java.io.File

/**
  * phuonglh, April, 2018; updated December 2019.
  *
  */
class W2V(spark: SparkSession, config: ConfigW2V) extends Serializable {
  final val logger = LoggerFactory.getLogger(getClass.getName)

  def train: Unit = {
    import spark.implicits._
    // data is one .txt file or one directory of .json files (result of TokenizerSparkApp)
    val dataset = if (config.text) {
        spark.sparkContext.textFile(config.input).filter(_.trim.size > config.minLength).toDF(config.inputCol)
    } else spark.read.json(config.input)
    dataset.cache()
    logger.info("#(texts) = " + dataset.count())
    dataset.show(20)

    val tokenizer = new Tokenizer().setInputCol(config.inputCol).setOutputCol("tokens")
    val w2v = new Word2Vec().setInputCol("tokens").setOutputCol("vector")
      .setMinCount(config.minFrequency).setVectorSize(config.dimension).setWindowSize(config.windowSize)
      .setMaxIter(config.iterations).setSeed(220712)
    val pipeline = new Pipeline().setStages(Array(tokenizer, w2v))
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
        val line = row.getAs[String](0) + " " + row.getAs[DenseVector](1).values.mkString(" ")
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
    Logger.getLogger("org.apache.spark").setLevel(Level.INFO)

    val parser = new OptionParser[ConfigW2V]("vlp.vec") {
      head("vlp.vec", "1.0")
      opt[String]('M', "master").action((x, conf) => conf.copy(master = x)).text("Spark master, default is local[*]")
      opt[String]('e', "executorMemory").action((x, conf) => conf.copy(master = x)).text("executor memory, default is 8g")
      opt[String]('m', "mode").action((x, conf) => conf.copy(mode = x)).text("running mode, either eval/train/test")
      opt[Unit]('v', "verbose").action((_, conf) => conf.copy(verbose = true)).text("verbose mode")
      opt[Int]('f', "minFrequency").action((x, conf) => conf.copy(minFrequency = x)).text("min feature frequency")
      opt[Int]('l', "minLength").action((x, conf) => conf.copy(minLength = x)).text("min sentence length in characters, default is 20")
      opt[Int]('w', "windowSize").action((x, conf) => conf.copy(windowSize = x)).text("windows size, default is 3")
      opt[Int]('d', "dimension").action((x, conf) => conf.copy(dimension = x)).text("vector dimension, default is 100")
      opt[Int]('k', "iterations").action((x, conf) => conf.copy(iterations = x)).text("number of iterations, default is 30")
      opt[String]('x', "inputCol").action((x, conf) => conf.copy(inputCol = x)).text("input column, default is 'text'")
      opt[String]('i', "input").action((x, conf) => conf.copy(input = x)).text("input data path")
      opt[Unit]('t', "textFormat").action((_, conf) => conf.copy(text = true)).text("text input data format instead of JSON")
      opt[String]('p', "modelPath").action((x, conf) => conf.copy(modelPath = x)).text("model path, default is 'dat/vec/'")
      opt[String]('o', "output").action((x, conf) => conf.copy(output = x)).text("output path")
    }
    parser.parse(args, ConfigW2V()) match {
      case Some(config) =>
        val sparkSession = SparkSession.builder().appName(getClass.getName).master(config.master)
          .config("spark.executor.memory", config.executorMemory).getOrCreate()
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
