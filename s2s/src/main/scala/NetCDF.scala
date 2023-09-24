import ucar.nc2.NetcdfFiles

object NetCDF {
  def main(args: Array[String]): Unit = {
    val path = "dat/y_tp_1.nc"
    val ncFile = NetcdfFiles.open(path)
    println(ncFile.getDetailInfo)
    ncFile.close()
  }
}
