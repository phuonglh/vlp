package vlp.frl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.BeforeAndAfter
import scala.util.Random

class GreedyAgentSpec extends AnyFlatSpec with BeforeAndAfter {
  val random = new Random(0)

  val qValues = Array[Float](0, 0, 0.5f, 0, 0)
  val armCount = Array[Int](0, 1, 0, 0, 0)
  val lastAction = 1
  val agent = GreedyAgent(random, qValues, armCount, lastAction)

  val qValues2 = Array[Float](0, 0, 1.0f, 0, 0)
  val armCount2 = Array[Int](0, 1, 0, 0, 0)
  val agent2 = GreedyAgent(random, qValues2, armCount2, lastAction)

  before {
    agent.step(reward = 1)
  }

  "GreedyAgent 1" should "update its values correctly" in {    
    val expectedValues = Array[Float](0, 0.5f, 0.5f, 0, 0)
    assert(agent.getValues() === expectedValues)
  }

  it should "take greedy action" in {
    assert(agent.getLastAction() == 2)
  }

  "GreedyAgent 2" should "behave correctly in two steps" in {
    agent2.step(reward = 1)
    assert(agent2.getLastAction() == 2)
    var expectedValues = Array[Float](0, 0.5f, 1.0f, 0, 0)
    assert(agent2.getValues() === expectedValues)

    agent2.step(reward = 2)
    assert(agent2.getLastAction() == 2)
    expectedValues = Array[Float](0, 0.5f, 2.0f, 0, 0)
    assert(agent2.getValues() === expectedValues)
  }
}
