package vlp.con

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, IdentityOutputShape, TensorModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.utils.{Shape, Table, T}
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
  * A customized layer that squezes all tensors inside a table. It is suppose that 
  * all the elements in the table is a 3-d tensor of shape bx1xd, where b is the 
  * batch size and d is the domain dimension. We need to squeeze each element to 
  * the shape bxd.
  * 
  * @author phuonglh@gmail.com
  * <p/> December 2022
  *
  * @param ev
  */
class SqueezeTable[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Activity, T] {

  override def updateOutput(input: Table): Activity = {
    val x = input(1).asInstanceOf[Tensor[T]]
    val b = x.size.head // the batch size 
    val d = x.size.last // the domain dimension
    val result = T()
    for (i <- 1 to input.length) { // for BERT input, this length is 4.
      val inp = input(i).asInstanceOf[Tensor[T]]
      val out = Tensor[T](inp.size) // zero tensor of the same size as inp
      result(i) = out.copy(inp).squeeze() // create a copy so as to keep inp unchanged
    }
    result
  }

  override def updateGradInput(input: Table, gradOutput: Activity): Table = {
    gradOutput.toTable
  }  
}

object SqueezeTable {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): SqueezeTable[T] = new SqueezeTable()
}

/**
  * Keras-style layer of the layer
  *
  * @param inputShape
  * @param ev
  */

class SqueezeTableLayer[T: ClassTag](val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Table, Table, T](KerasUtils.addBatch(inputShape)) 
    with IdentityOutputShape with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Table, Table, T] = {
    val layer = SqueezeTable()
    layer.asInstanceOf[AbstractModule[Table, Table, T]]
  }
}

object SqueezeTableLayer {
  def apply[@specialized(Float, Double) T: ClassTag](inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SqueezeTableLayer[T] = {
    new SqueezeTableLayer[T](inputShape)
  }
}