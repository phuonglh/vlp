package vlp.con

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
  * A customized ArgMax layer that operates on tensor rather than on Table of the BigDL lib. 
  * Currently, the argmax on the last dimension is assumed. Note that the index is started from 1 in BigDL.
  * 
  * @author phuonglh@gmail.com
  *
  * @param ev
  */
class ArgMax[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dimension = input.size.length
    val (_, result) = input.asInstanceOf[Tensor[NumericWildcard]].max(dimension)
    output.resizeAs(result)
    result.cast[T](output)
    output.squeeze(dimension)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradOutput
  }  
}

object ArgMax {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): ArgMax[T] = new ArgMax()
}

/**
  * Keras-style layer of the ArgMax
  *
  * @param inputShape
  * @param ev
  */

class ArgMaxLayer[T: ClassTag](val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[Int], T](KerasUtils.addBatch(inputShape)) with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[Int], T] = {
    val layer = ArgMax()
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[Int], T]]
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray 
    Shape(input.slice(0, input.size - 1)) // don't take the last dimension
  }
}

object ArgMaxLayer {
  def apply[@specialized(Float, Double) T: ClassTag](inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ArgMaxLayer[T] = {
    new ArgMaxLayer[T](inputShape)
  }
}