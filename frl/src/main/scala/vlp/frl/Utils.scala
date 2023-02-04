package vlp.frl
import scala.util.Random
import scala.collection.mutable.ArrayBuffer


object Utils {
  val zeroSeedRandom = new Random(0)
  /**
    * Takes in a list of values and returns the index of the item 
    * with the highest value. Breaks ties randomly.
    * @param values
    * @param random 
    * @return int - the index of the highest value in q_values
    */
  def argmax(values: Array[Float], random: Random): Int = {
    var topValue = Float.NegativeInfinity
    val ties = ArrayBuffer[Int]()
    for (i <- 0 until values.size) 
      if (values(i) > topValue) topValue = values(i)
    for (i <- 0 until values.size) 
      if (values(i) == topValue) ties.append(i)
    ties(random.nextInt(ties.size))
  }

  def argmax(values: Array[Float]): Int = {
    argmax(values, zeroSeedRandom)
  }
  
}
