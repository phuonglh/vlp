package vlp.ner

import org.apache.spark.sql.DataFrame

/**
  * A simple label-based evaluator. Given a data frame having 3 columns "x, y, z" where 
  * x contains input sequences, y contains correct label sequences and z contains predicted label sequences, 
  * we compute accuracy scores.
  */

object Evaluator {
  def run(df: DataFrame): Map[String, Double] = {
    import df.sparkSession.implicits._
    val result = df.select("words", "y", "z").flatMap(row => {
      val n = row.getAs[Seq[String]](0).size
      val ys = row.getAs[String](1).split(" ")
      val zs = row.getAs[Seq[String]](2).take(n)
      ys.zip(zs).map{ case (y, z) => (y, z.toUpperCase()) }
    }).collect()
    val scores = result.groupBy(_._1).map { case (k, pairs) => 
      val m = pairs.filter(_._2 == k)
      (k, m.size.toDouble/pairs.size)
    }
    scores
  }
}