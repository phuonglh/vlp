package vlp.frl

import org.scalatest.flatspec.AnyFlatSpec


class ArgmaxSpec extends AnyFlatSpec {

  "Argmax" should "return the index of the largest value" in {
    val values = Array[Float](0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
    assert(Utils.argmax(values) == 8)
  }

  it should "call random choice correctly" in {
    val values = Array[Float](1, 0, 0, 1)
    assert(Utils.argmax(values) == 0)
  }

  it should "does not always choose the first entry or the last entry" in {
    val values = Array[Float](1, 0, 0, 1)
    val count = Array.fill[Int](4)(0)
    for (_ <- 0 until 100) {
      val i = Utils.argmax(values)
      count(i) = count(i) + 1
    }
    assert(count(0) != 100)
    assert(count(3) != 100)
    val expected = Array(47, 0, 0, 53)
    // note the triple-equal operator to compare content of two arrays
    assert(count === expected) 
  }
}
