package vlp.vdg

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.optim.{AccuracyResult, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

/**
  * phuonglh
  * @param ev
  */
class TimeDistributedTop1Accuracy(paddingValue: Int = -1)(implicit ev: TensorNumeric[Float]) extends ValidationMethod[Float] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    var correct = 0
    var count = 0
    val _output = output.asInstanceOf[Tensor[Float]]
    val _target = target.asInstanceOf[Tensor[Float]]
    _output.split(1).zip(_target.split(1)).foreach { case (tensor, ys) => 
      val zs = tensor.split(1).map { t =>
        val values = t.toArray()
        val k = (0 until values.size).zip(values).maxBy(p => p._2)._1
        k + 1 // one-based label index
      }
      // filter the padded value in the gold target before matching
      // with the prediction
      val c = ys.toArray().filter(e => e != paddingValue).zip(zs)
        .map(p => if (p._1 == p._2) 1 else 0)
      correct += c.sum
      count += c.size
    }
    new AccuracyResult(correct, count)
  }
  override def format(): String = "Time Distributed Top-1 Accuracy"
}
