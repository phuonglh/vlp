import ucar.ma2.Range
import ucar.nc2.NetcdfFiles
import ucar.nc2.write.Ncdump

/**
 * Vietnam: 8:24 N, 102:110 E. On the grid of 2.5 x 2.5, the index
 * of Vietnam is 26:33 of latitude (which ranges from 90 to -90) and 40:44 of longitude (which ranges from 0 to 360).
 */
object NetCDF {
  val stations = Array((22.366, 102.833), (22.366, 103.233), (22.416, 103.4833), (21.95, 103.883), (22.0666, 103.15))
  /**
   * Converts (latitude, longitude) to a pairs of indices (i, j)
   * @param lon
   * @param lat
   * @return (latId, lonId)
   */
  def index(lat: Double, lon: Double): (Int, Int) = {
    val latId = ((90-lat)/2.5).toInt
    val lonId = (lon/2.5).toInt
    (latId, lonId)
  }

  def readSLP(path: String): String = {
    val ncFile = NetcdfFiles.open(path)
    println(ncFile.getDetailInfo)
    val variables = ncFile.getVariables
    variables.forEach(println(_))
    // read the slp variable
    val slp = ncFile.findVariable("slp")
    // filter the Vietnam region
    val data = slp.read(":,26:33,40:44") // [time, lat, lon]
    val st = Ncdump.printArray(data, "slp", null)
    println("Shape of data = " + data.getShape.mkString(" x "))
    ncFile.close()
    st
  }

  def readSLP(path: String, lat: Double, lon: Double): List[Float] = {
    val ncFile = NetcdfFiles.open(path)
    println(ncFile.getDetailInfo)
    val variables = ncFile.getVariables
    variables.forEach(println(_))
    // read the slp variable
    val slp = ncFile.findVariable("slp")
    // filter the Vietnam region
    val (latId, lonId) = index(lat, lon)
    val data = slp.read(s":,$latId,$lonId").reduce() // [time, lat, lon]
//    val st = Ncdump.printArray(data, "slp", null)
//    println("Shape of data = " + data.getShape.mkString(" x "))
    ncFile.close()
    val xs = collection.mutable.ListBuffer[Float]()
    for (i <- 0 until data.getSize.toInt)
      xs += data.getFloat(i)
    xs.toList
  }

  /**
   * Extracts two geopotential levels at H850 and H500 at every day. List[(H850, H500)].
   * @param path
   * @param lat
   * @param lon
   * @return a list of 366 elements, each element is a pair.
   */
  def readHGT(path: String, lat: Double, lon: Double): List[(Float, Float)] = {
    // there are 17 levels
    // 1000 925 850 700 600 500 400 300 250 200 150 100 70 50 30 20 10
    val ncFile = NetcdfFiles.open(path)
    println(ncFile.getDetailInfo)
    val variables = ncFile.getVariables
    variables.forEach(println(_))
//    // read the hgt variable
    val hgt = ncFile.findVariable("hgt")
    // filter at the specified position
    val (latId, lonId) = index(lat, lon)
//    val ranges1 = List(new Range(2, 2), new Range(latId, latId), new Range(lonId, lonId))
    // [time, 2, lat, lon] ==> reduce to an array of 366 elements
    val h850 = hgt.read(s":,2,$latId,$lonId").reduce()
//    val st850 = Ncdump.printArray(h850, "hgt", null)
    // [time, 5, lat, lon] ==> reduce to an array of 366 elements
    val h500 = hgt.read(s":,5,$latId,$lonId").reduce()
    val xs = collection.mutable.ListBuffer[(Float, Float)]()
    for (i <- 0 until h850.getSize.toInt) {
      xs += ((h850.getFloat(i), h500.getFloat(i)))
    }
    ncFile.close()
    xs.toList
  }

  def main(args: Array[String]): Unit = {
    val ids = stations.foreach(p => print(index(p._1, p._2)))

//    val path = "dat/slp/slp.1980.nc"
//    val xs = readSLP(path, stations.head._1, stations.head._2)

    val path = "dat/geo/hgt.1980.nc"
    val xs = readHGT(path, stations.head._1, stations.head._2)
    println("Length of xs = " + xs.size)
    println(xs.mkString(", "))
  }
}
