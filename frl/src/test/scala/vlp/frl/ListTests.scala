package vlp.frl

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.flatspec.AnyFlatSpec

// use FunSuite testing style
class ListFunSuite extends AnyFunSuite {
  test("An empty list should have size 0") {
    assert(List.empty.size == 0)
  }

  test("Accessing invalid index should throw IndexOutOfBoundsException") {
    val fruit = List("Banana", "Pineapple", "Apple")
    assert(fruit.head == "Banana")
    assertThrows[IndexOutOfBoundsException] {
      fruit(5)
    }
  }  
}

// use FlatSpec testing style (for BDD style testing)
class ListFlatSpec extends AnyFlatSpec {

  "An empty List" should "have size 0" in {
    assert(List.empty.size == 0)
  }

  it should "throw an IndexOutOfBoundsException when trying to access any element" in {
    val emptyList = List()
    assertThrows[IndexOutOfBoundsException] {
      emptyList(1)
    }
  }

}