package vlp.con

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.optim.{AccuracyResult, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

/**
  * phuonglh@gamil.com
  * 
  * @param paddingValue
  * @param ev
 *
 * Note: 1-based label index for token classification
  */
class TimeDistributedTop1Accuracy(paddingValue: Int = -1)(implicit ev: TensorNumeric[Float]) extends ValidationMethod[Float] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    var correct = 0
    var count = 0
    val _output = output.asInstanceOf[Tensor[Float]] // nDim = 3
    val _target = target.asInstanceOf[Tensor[Float]] // nDim = 2
    // split by batch size (dim = 1 of output and target)
    _output.split(1).zip(_target.split(1))
      .foreach { case (tensor, ys) =>
      // split by time slice (dim = 1 of tensor)
      val zs = tensor.split(1).map { t =>
        val (_, k) = t.max(1) // the label with max score
        k(Array(1)).toInt // k is a tensor => extract its value
      }
//      println(zs.mkString(", ") + " :=: " + ys.toArray().mkString(", ")) // DEBUG
      // filter the padded value (-1f) in the target before perform matching with the output
      val c = ys.toArray().map(_.toInt).filter(e => e != paddingValue).zip(zs)
        .map(p => if (p._1 == p._2) 1 else 0)
      correct += c.sum
      count += c.size
    }
    new AccuracyResult(correct, count)
  }
  override def format(): String = "TimeDistributedTop1Accuracy"
}
