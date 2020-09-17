package vlp.ner

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.feature.{HashingTF, StringIndexerModel}
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

object DecoderType extends Enumeration {
  val Greedy, Viterbi = Value 
}

/**
 * @author Phuong LE-HONG
 * <p>
 * Sep 20, 2016, 5:14:41 PM
 * <p>
 * Sequence decoder. 
 * 
 */
class Decoder(spark: SparkSession, decodeType: DecoderType.Value, model: PipelineModel, extendedFeatureSet: Set[String]) {
  val logger = LoggerFactory.getLogger(getClass)
  val labels = model.stages(0).asInstanceOf[StringIndexerModel].labels
  val numLabels = labels.length
  val numFeatures = model.stages(2).asInstanceOf[HashingTF].getNumFeatures
  val hashingTF = new org.apache.spark.mllib.feature.HashingTF(numFeatures)
  val (weights, bias) = coefficients

  /**
   * Loads the pipeline model and extract parameter values of the main 
   * statistical classifier.
   * @return parameter values in form of a pair (weight matrix, bias vector)
   */
  def coefficients: (Matrix, Vector) = {
    val logreg = model.stages(3).asInstanceOf[LogisticRegressionModel]
    logger.info("shape(theta) = " + logreg.coefficientMatrix.numRows + " x " + logreg.coefficientMatrix.numCols)
    logger.info("length(beta) = " + logreg.interceptVector.size)
    (logreg.coefficientMatrix, logreg.interceptVector)
  }
  
  /**
   * Decodes a list of sentences.
   * @param sentences
   * @return a list of decoded sentences.
   */
  def decode(sentences: List[Sentence]): List[Sentence] = {
    sentences.par.map(s => decode(s)).toList
  }

  /**
   * Decodes a sentence.
   * @param sentence
   * @return in-place updated sentence.
   */
  def decode(sentence: Sentence): Sentence = {
    def greedy(): Sentence = {
      val tokens = sentence.tokens
      val numLabels = labels.length
      for (j <- 0 until tokens.length) {
        val context = Featurizer.extract(sentence, j, extendedFeatureSet)
        val features = context.bof.toLowerCase().split("\\s+")
        val x = hashingTF.transform(features).toSparse      
        val scores = (0 until numLabels).map {
          k => {
            x.indices.map(j => weights(k, j)).sum + bias(k)
          }
        }.toList
        val bestLabel = scores.zipWithIndex.maxBy(_._1)._2
        val kv = (Label.NamedEntity, labels(bestLabel))
        tokens.update(j, Token(tokens(j).word, tokens(j).annotation + kv))
      }
      sentence
    }
    
    def viterbi(): Sentence = {
      val tokens = sentence.tokens
      val numLabels = labels.length
      val lattice = Array.ofDim[Double](numLabels, tokens.length) 
      // compute lattice
      for (j <- 0 until tokens.length) {
        val context = Featurizer.extract(sentence, j, extendedFeatureSet)
        val features = context.bof.toLowerCase().split("\\s+")
        val x = hashingTF.transform(features).toSparse
        val scores = (0 until numLabels).map {
          k => {
            x.indices.map(j => weights(k, j)).sum + bias(k)
          }
        }
        // assign the best guess for the current tag for the next feature extraction
        // this is a rather greedy approach.
        val bestLabel = scores.zipWithIndex.maxBy(_._1)._2
        val kv = (Label.NamedEntity, labels(bestLabel))
        tokens.update(j, Token(tokens(j).word, tokens(j).annotation + kv))
        // update column j of the lattice
        val max = scores.max
        val total = scores.map(s => math.pow(math.E, s - max)).sum
        val logTotal = math.log(total)
        val logProbs = scores.map(s => s - max - logTotal)
        for (k <- 0 until numLabels) {
          lattice(k)(j) = logProbs(k)
        }
      }
      // find best path
      val bestPath = Viterbi.decode(lattice)._1
      // update result
      for (j <- 0 until tokens.length) {
        val kv = (Label.NamedEntity, labels(bestPath(j)))
        tokens.update(j, Token(tokens(j).word, tokens(j).annotation + kv))
      }
      sentence
    }

    decodeType match {
      case DecoderType.Greedy => greedy()
      case DecoderType.Viterbi => viterbi()
    }

  }
  
}
