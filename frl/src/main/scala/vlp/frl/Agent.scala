package vlp.frl
import scala.util.Random

abstract class Agent(numArms: Int) {
  protected var qValues: Array[Float] = Array.fill[Float](numArms)(0f)
  protected var armCount: Array[Int] = Array.fill[Int](numArms)(0)
  protected var lastAction: Int = -1

  /**
    * Takes one step for the agent. It takes in a reward and observation and returns the action 
    * the agent chooses at that time step.
    *
    * @param reward float, the reward the agent recieved from the environment after taking the last action.        
    * @param obversation float, the observed state the agent is in.
    * @return current action -- int, the action chosen by the agent at the current time step.
    */
  def step(reward: Float, obversation: Option[Any]=None): Int

  def getValues(): Array[Float] = qValues
  def getLastAction(): Int = lastAction
}


class GreedyAgent(numArms: Int, random: Random = new Random(0)) extends Agent(numArms) {

  def this(random: Random, qValues: Array[Float], armCount: Array[Int], lastAction: Int) = {
    this(qValues.size, random)
    this.qValues = qValues
    this.armCount = armCount
    this.lastAction = lastAction
  }


  def step(reward: Float, obversation: Option[Any]): Int = {
    armCount(lastAction) += 1
    qValues(lastAction) += (reward - qValues(lastAction))/armCount(lastAction)
    val currentAction = Utils.argmax(qValues, random)
    lastAction = currentAction    
    return currentAction
  }
}

object GreedyAgent {

  def apply(random: Random, qValues: Array[Float], armCount: Array[Int], lastAction: Int): GreedyAgent = {
    new GreedyAgent(random, qValues, armCount, lastAction)
  }

  def apply(qValues: Array[Float], armCount: Array[Int], lastAction: Int): GreedyAgent = {
    new GreedyAgent(new Random(0), qValues, armCount, lastAction)
  }
}