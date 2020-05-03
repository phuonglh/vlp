package vlp

/**
  * A package object containing some utility functions.
  */

package object VLP {
  def log(message: String): Unit = {
    println(s"${Thread.currentThread.getName}: $message")
  }

  def timing(body: => Unit) = {
    val start = System.nanoTime()
    body
    val end = System.nanoTime()
    val duration = ((end - start) / 1000) / 1000.0
    println(s"duration = ${duration} (milliseconds)")
  }
}
