package vlp.woz.act

import com.intel.analytics.bigdl.dllib.tensor.{IntType, Tensor}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
  * A customized layer that operates on tensor to extract top labels from a prediction vector. 
  * NOTE: This utility is not used anymore. It has been replaced by [[TopKSelector]] transformer.
  * 
  * @author phuonglh@gmail.com
  * @param k: 
  * @param ev
  */
class TopK[T: ClassTag](k: Int = 1)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dimension = input.size.length
    val p = input.asInstanceOf[Tensor[Float]]
    // sort by values in decreasing order
    val indices = p.toArray.zipWithIndex.sortBy(-_._1).take(k).map(_._2)
    val result = Tensor[Float](indices.size)
    for (j <- 1 to indices.size) 
      result.setValue(j, indices(j-1).toFloat)
    output.resizeAs(result)
    result.cast[T](output)
    output.squeeze(dimension)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradOutput
  }  
}

object TopK {
  def apply[T: ClassTag](k: Int = 1)(implicit ev: TensorNumeric[T]): TopK[T] = new TopK(k)
}

/**
  * Keras-style layer
  *
  * @param inputShape
  * @param ev
  */

class TopKLayer[T: ClassTag](val k: Int = 1, val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[Int], T](KerasUtils.addBatch(inputShape)) with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[Int], T] = {
    val layer = TopK(k)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[Int], T]]
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray 
    Shape(input.slice(0, input.size - 1)) // don't take the last dimension
  }
}

object TopKLayer {
  def apply[@specialized(Float, Double) T: ClassTag](k: Int = 1, inputShape: Shape = null)(implicit ev: TensorNumeric[T]): TopKLayer[T] = {
    new TopKLayer[T](k, inputShape)
  }
}