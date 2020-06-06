package vlp.vdg

import com.intel.analytics.bigdl.nn.MapTable
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.serializer.{ContainerSerializable, DeserializeContext, SerializeContext}

import scala.reflect.ClassTag

/**
  * phuonglh
  *
  */
@SerialVersionUID(4403280698280280273L)
class MapTableNoAcc[T: ClassTag](val m: AbstractModule[_ <: Activity, _ <: Activity, T] = null)(implicit ev: TensorNumeric[T]) extends MapTable(m) {

  if (module != null) {
    this.add(module)
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    // empty
  }

  override def skipDuplicateCheck = true

  override def toString(): String = {
    val tab = "  "
    val line = "\n"
    var str = s"${getPrintName}"
    if (module != null) {
      str += s"{$line$tab$module$line}"
    } else {
      str += " { }"
    }
    str
  }
}

object MapTableNoAcc extends ContainerSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](module: AbstractModule[_ <: Activity, _ <: Activity, T] = null)(implicit ev: TensorNumeric[T]) : MapTableNoAcc[T] = {
    new MapTableNoAcc[T](module)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)(implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val mapTable = super.doLoadModule(context).asInstanceOf[MapTableNoAcc[T]]
    require(mapTable.modules.size >=1, "sub module should not be empty")
    mapTable.add(mapTable.modules(0))
    mapTable
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T], mapBuilder : BigDLModule.Builder)(implicit ev: TensorNumeric[T]) : Unit = {
    val mapTable = context.moduleData.module.asInstanceOf[MapTableNoAcc[T]]
    val subModules = mapTable.modules
    require(subModules.size >=1, "sub module should not be empty")
    // `modules` are created during forward() by 'n' times of the same module depends on input size,
    // store the first one to save the storage cost just in case large input size
    val singleModule = subModules(0)
    mapTable.modules.clear()
    mapTable.modules.append(singleModule)
    super.doSerializeModule(context, mapBuilder)
  }
}
