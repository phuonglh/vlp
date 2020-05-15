package vlp.vdr

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.udf
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.collection.mutable.ListBuffer


trait DecoderParams extends Params {
  final val token: BooleanParam = new BooleanParam(this, "token-based", "use syllable segmentation rather than vowel segmentation")
  def isToken: Boolean = $(token)
  def setToken(value: Boolean): this.type = set(token, value)
  final val greedy: BooleanParam = new BooleanParam(this, "greedy decoding", "use greedy decoding instead of Viterbi decoding")
  def isGreedy: Boolean = $(greedy)
  def setGreedy(value: Boolean): this.type = set(greedy, value)
  
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  
  setDefault(token -> true, greedy -> false, inputCol -> "x", outputCol -> "y")
} 

/**
  * phuonglh, 11/1/17, 18:20
  * <p>
  *   Sequence decoder with MLR local models. 
  */
class Decoder(override val uid: String,
              val labels: Array[String],
              val featureTypes: Array[String], val markovOrder: Int, val numFeatures: Int,
              val weights: Matrix, val intercepts: Vector)
  extends Transformer with DecoderParams with DefaultParamsWritable {
  
  final val logger = LoggerFactory.getLogger(getClass.getName)
  
  val hashingTF = new HashingTF(numFeatures)

  val topK: IntParam = new IntParam(this, "topK", "topK", ParamValidators.gt(0))
  def setTopK(value: Int): this.type = set(topK, value)
  def getTopK: Int = $(topK)
  
  setDefault(topK -> 5)
  
  def this(labels: Array[String], featureTypes: Array[String], markovOrder: Int, numFeatures: Int, weights: Matrix, intercepts: Vector) =
    this(Identifiable.randomUID("decoder"), labels, featureTypes, markovOrder, numFeatures, weights, intercepts)

  protected def createTransformFunc: (String) => Array[String] = { (input: String) =>
    /**
      * Finds the best accented solution for a given un-accented text x.
      * This function treats the input as a sequence of characters:
      * "toi la giang vien" ==> ["t", "o", "i", " ", "l", "a",..., "ie", "n"].
      * This function uses a greedy decoding algorithm.
      */
    def f(x: String): String = {
      val ys = new ListBuffer[String]
      val xs = Sequencer.transform(x, false)
      for (j <- 0 until xs.size) {
        if (Decoder.trigger(xs(j))) {
          val fs = Sampler.extract(xs, ys, featureTypes, markovOrder, j)
          val is = hashingTF.transform(fs).toSparse
          val candidates = xs(j).size match {
            case 1 => VieMap.singles(xs(j))
            case 2 => VieMap.pairs(xs(j))
            case 3 => VieMap.triples(xs(j))
            case _ => throw new IllegalArgumentException("It is impossible to have such a case!")
          }
          val ls = candidates.filter(labels.contains(_))
          val scores = (0 until ls.size).map(k => {
            val u = labels.indexOf(ls(k))
            is.indices.map(weights(u, _)).sum + intercepts(u)
          }).toList
          val maxId = scores.zipWithIndex.maxBy(_._1)._2
          val bestCandidate = ls(maxId)
          ys.append(bestCandidate)
        } else ys.append(xs(j))
      }
      ys.mkString("").trim
    }

    /**
      * Finds the best accented solution for a given un-accented text x.
      * This function treats the input as a sequence of syllables.
      * "toi la giang vien" ==> ["toi", "la", "giang", "vien"]
      * @param x a string such as "toi la giang vien"
      * @return a string: "tôi là giảng viên"
      */
    def g(x: String): String = {
      val ys = new ListBuffer[String]
      val xs = Sequencer.transform(x, false)
      val ps = mutable.Map[Int, List[String]]()
      // collect candidates at each position
      //
      for (j <- 0 until xs.size) {
        if (Decoder.trigger(xs(j))) {
          val fs = Sampler.extract(xs, ys, featureTypes, markovOrder, j)
          val is = hashingTF.transform(fs).toSparse
          val candidates = xs(j).size match {
            case 1 => VieMap.singles(xs(j))
            case 2 => VieMap.pairs(xs(j))
            case 3 => VieMap.triples(xs(j))
            case _ => throw new IllegalArgumentException("It is impossible to have such a case!")
          }
          val ls = candidates.filter(labels.contains(_))
          val scores = (0 until ls.size).map(k => {
            val u = labels.indexOf(ls(k))
            is.indices.map(weights(u, _)).sum + intercepts(u)
          }).toList
          val maxId = scores.zipWithIndex.maxBy(_._1)._2
          val bestCandidate = ls(maxId)
          ys.append(bestCandidate)
          val topCandidates = scores.zipWithIndex.sortBy(_._1)(Ordering[Double].reverse).take(3).map(_._2).map(ls(_))
          ps += (j -> topCandidates)
        } else ys.append(xs(j))
      }
      // filter the invalid candidates
      for (j <- 0 until xs.size) {
        if (Decoder.trigger(xs(j))) {
          var u = j - 1
          while (u >= 0 && !Lexicon.punctuations.contains(xs(u))) u = u - 1
          var v = j + 1
          while (v < xs.size && !Lexicon.punctuations.contains(xs(v))) v = v + 1
          val left = xs.slice(u+1, j)
          val right = xs.slice(j+1, v)
          val syllables = ps(j).map { e => 
            val syllable = (left ++ List(e) ++ right).mkString
            (j, e, syllable)
          }.filter(t => Decoder.isValid(t._3))
          if (syllables.nonEmpty) {
            val bestCandidate = syllables.head._2
            ys(j) = bestCandidate
          }
        }
      }
      ys.mkString("").trim
    }

    /**
      * Viterbi search function which improves over g(.) decoding function.
      * @param x a string ("toi la giang vien")
      * @return a string ("tôi là giảng viên")
      */
    def v(x: String): String = {
      val ys = new ListBuffer[String]
      val xs = Sequencer.transform(x, false)
      val ps = mutable.Map[Int, List[(String, Double)]]() // position j -> List[(label, logProb)]
      // collect candidates at each position
      //
      val beamSize = $(topK)
      val lattice = Array.ofDim[Double](beamSize, xs.size)
      for (i <- 0 until beamSize; j <- 0 until xs.size)
        lattice(i)(j) = Double.NegativeInfinity
      for (j <- 0 until xs.size) {
        if (Decoder.trigger(xs(j))) {
          val fs = Sampler.extract(xs, ys, featureTypes, markovOrder, j)
          val is = hashingTF.transform(fs).toSparse
          val candidates = xs(j).size match {
            case 1 => VieMap.singles(xs(j))
            case 2 => VieMap.pairs(xs(j))
            case 3 => VieMap.triples(xs(j))
            case _ => throw new IllegalArgumentException("Decoder: It is impossible to have such a case!")
          }
          val ls = candidates.filter(labels.contains(_))
          val scores = (0 until ls.size).map(k => {
            val u = labels.indexOf(ls(k))
            is.indices.map(weights(u, _)).sum + intercepts(u)
          }).toList
          val maxScore = scores.max
          val adjustScores = scores.map(s => s - maxScore)
          val z = adjustScores.map(s => Math.exp(s)).sum
          val normalizedScores = adjustScores.map(s => s - Math.log(z))
          val maxId = normalizedScores.zipWithIndex.maxBy(_._1)._2
          val bestCandidate = ls(maxId)
          ys.append(bestCandidate)
          val topCandidates = normalizedScores.zipWithIndex.sortBy(_._1)(Ordering[Double].reverse).take(beamSize)
          ps += (j -> topCandidates.map(p => (ls(p._2), p._1)))
          for (i <- 0 until topCandidates.size)
            lattice(i)(j) = topCandidates(i)._1
        } else {
          ys.append(xs(j))
          for (i <- 0 until beamSize)
            lattice(i)(j) = 0d
        }
      }
      val path = Viterbi.decode(lattice)._1
      val result = new ListBuffer[String]
      for (j <- 0 until xs.size) {
        if (ps.contains(j)) {
          val topLabels = ps(j).map(_._1)
          result.append(topLabels(path(j)))
        } else result.append(ys(j))
      }
      result.mkString("").trim
    }
    
    /**
      * Beam search for multiple candidate solutions. (Per requested by VCM, March 2018).
      * @param x an input un-accented sequence
      * @return a list of accented sequences
      */
    def b(x: String): List[String] = {
      val ys = new ListBuffer[String]
      val xs = Sequencer.transform(x, false)
      val ps = mutable.Map[Int, List[(String, Double)]]() // position j -> List[(label, logProb)]
      // collect candidates at each position
      //
      val beamSize = $(topK)
      val lattice = Array.ofDim[Double](beamSize, xs.size)
      for (i <- 0 until beamSize; j <- 0 until xs.size)
        lattice(i)(j) = Double.NegativeInfinity
      var n = 0
      val js = mutable.ListBuffer[Int]()
      val jm = mutable.Map[Int, Int]()
      for (j <- 0 until xs.size) {
        if (Decoder.trigger(xs(j))) {
          val fs = Sampler.extract(xs, ys, featureTypes, markovOrder, j)
          val is = hashingTF.transform(fs).toSparse
          val candidates = xs(j).size match {
            case 1 => VieMap.singles(xs(j))
            case 2 => VieMap.pairs(xs(j))
            case 3 => VieMap.triples(xs(j))
            case _ => throw new IllegalArgumentException("Decoder: It is impossible to have such a case!")
          }
          val ls = candidates.filter(labels.contains(_))
          val scores = (0 until ls.size).map(k => {
            val u = labels.indexOf(ls(k))
            is.indices.map(weights(u, _)).sum + intercepts(u)
          }).toList
          val maxScore = scores.max
          val adjustScores = scores.map(s => s - maxScore)
          val z = adjustScores.map(s => Math.exp(s)).sum
          val normalizedScores = adjustScores.map(s => s - Math.log(z))
          val maxId = normalizedScores.zipWithIndex.maxBy(_._1)._2
          val bestCandidate = ls(maxId)
          ys.append(bestCandidate)
          val topCandidates = normalizedScores.zipWithIndex.sortBy(_._1)(Ordering[Double].reverse).take(beamSize)
          ps += (j -> topCandidates.map(p => (ls(p._2), p._1)))
          for (i <- 0 until topCandidates.size)
            lattice(i)(j) = topCandidates(i)._1
          js += j
          jm += (j -> n)
          n = n + 1
        } else {
          ys.append(xs(j))
          for (i <- 0 until beamSize)
            lattice(i)(j) = 0d
        }
      }
      // scores is lattices with zero-columns removed
      val scores = Array.ofDim[Double](beamSize, n)
      for (r <- 0 until beamSize; c <- 0 until n)
        scores(r)(c) = lattice(r)(js(c))
      // beam search on scores
      val paths = BeamSearch.decode(scores, beamSize)
      val results = new ListBuffer[String]
      paths.map(_._1).foreach { path =>
        val result = new ListBuffer[String]
        for (j <- 0 until xs.size) {
          if (ps.contains(j)) {
            val topLabels = ps(j).map(_._1)
            val idx = math.min(path(jm(j)), topLabels.size-1)
            result.append(topLabels(idx))
          } else result.append(ys(j))
        }
        val solution = result.mkString("").trim
        results.append(solution)
      }
      results.toList
    }
    
    if (isGreedy) {
      if (!isToken) Array(f(input)); else Array(g(input)) 
    } else {
      if ($(topK) == 1) Array(v(input)) else b(input).toArray
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val transformUDF = udf(this.createTransformFunc, ArrayType(StringType, false))
    dataset.withColumn($(outputCol), transformUDF(dataset($(inputCol))))
  }

  override def copy(extra: ParamMap) = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    validateInputType(inputType)
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputFields = schema.fields :+
      StructField($(outputCol), new ArrayType(StringType, false), nullable = false)
    StructType(outputFields)
  }

  protected def validateInputType(inputType: DataType): Unit = {}
  
}

object Decoder extends DefaultParamsReadable[Decoder] {
  
  def trigger(s: String): Boolean = {
    VieMap.triples.keySet.contains(s) || VieMap.pairs.keySet.contains(s) || VieMap.singles.keySet.contains(s) 
  }
  
  def isValid(syllable: String): Boolean = {
    Lexicon.syllables.contains(syllable.toLowerCase)
  }
  
  override def load(path: String): Decoder = super.load(path)
}

