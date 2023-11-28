import ucar.nc2.NetcdfFiles
import ucar.nc2.write.Ncdump

import java.nio.file.{Files, Path, StandardOpenOption}

/**
 * Vietnam: 8:24 N, 102:110 E. On the grid of 2.5 x 2.5, the index
 * of Vietnam is 26:33 of latitude (which ranges from 90 to -90) and 40:44 of longitude (which ranges from 0 to 360).
 *
 * <p>phuonglh@gmail.com
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

  /**
   * Extracts the humidity at a position (lon, lat) at some pressure levels for every day
   * in a year. The year is specified in the given path.
   * @param path
   * @param lat
   * @param lon
   * @return a list of 366 arrays, each array contains humidity values at level 0 (1000mb), 2 (850mb), and 5 (500mb).
   */
  def readHumidity(path: String, lat: Double, lon: Double): List[Array[Float]] = {
    val ncFile = NetcdfFiles.open(path)
//    println(ncFile.getDetailInfo())
    val rhum = ncFile.findVariable("rhum")
    val (latId, lonId) = index(lat, lon)
    val h1000 = rhum.read(s":,0,$latId,$lonId").reduce() // [time, 0, lat, lon] ==> reduce to an array of 366 elements
    val h850 = rhum.read(s":,2,$latId,$lonId").reduce()
    val h500 = rhum.read(s":,5,$latId,$lonId").reduce()
    // collect values
    val xs = collection.mutable.ListBuffer[Array[Float]]()
    for (i <- 0 until h850.getSize.toInt) {
      xs += Array(h1000.getFloat(i), h850.getFloat(i), h500.getFloat(i))
    }
    ncFile.close()
    xs.toList

  }

  /**
   * Reads a directory of N years of data and extract daily humidity data at a pressure level.
   *
   * @param startYear
   * @param endYear
   * @param lat
   * @param lon
   * @return a list of N x 365 arrays, each array contains humidity data of a day at 3 pressure levels.
   */
  def readHumidity(startYear: Int, endYear: Int, lat: Double, lon: Double): List[Array[Float]] = {
    val paths = (startYear to endYear).map(year => s"dat/hum/rhum.$year.nc")
    val xs = paths.map(path => readHumidity(path, lat, lon))
    xs.reduce(_ ++ _)
  }

  def readSLP(path: String, lat: Double, lon: Double): List[Float] = {
    val ncFile = NetcdfFiles.open(path)
//    println(ncFile.getDetailInfo)
//    val variables = ncFile.getVariables
//    variables.forEach(println(_))
    // read the slp variable
    val slp = ncFile.findVariable("slp")
    // filter values at the station
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

  def readSLP(startYear: Int, endYear: Int, lat: Double, lon: Double): List[Float] = {
    val paths = (startYear to endYear).map(year => s"dat/slp/slp.$year.nc")
    paths.flatMap(path => readSLP(path, lat, lon)).toList
  }

  /**
   * Extracts two geopotential levels at H850 and H500 at every day. List[[H850, H500]].
   * @param path
   * @param lat
   * @param lon
   * @return a list of 366 elements, each element is a pair.
   */
  def readHGT(path: String, lat: Double, lon: Double): List[Array[Float]] = {
    // there are 17 levels: [1000 925 850 700 600 500 400 300 250 200 150 100 70 50 30 20 10]
    val ncFile = NetcdfFiles.open(path)
    // read the hgt variable
    val hgt = ncFile.findVariable("hgt")
    // filter values at the specified position
    val (latId, lonId) = index(lat, lon)
    val h850 = hgt.read(s":,2,$latId,$lonId").reduce() // [time, 2, lat, lon] ==> reduce to an array of 366 elements
//    val st850 = Ncdump.printArray(h850, "hgt", null)
    // [time, 5, lat, lon] ==> reduce to an array of 366 elements
    val h500 = hgt.read(s":,5,$latId,$lonId").reduce()
    // collect values
    val xs = collection.mutable.ListBuffer[Array[Float]]()
    for (i <- 0 until h850.getSize.toInt) {
      xs += Array(h850.getFloat(i), h500.getFloat(i))
    }
    ncFile.close()
    xs.toList
  }

  /**
   * Reads a directory of N years of data and extract daily HGT data.
   * @param startYear
   * @param endYear
   * @param lat
   * @param lon
   * @return a list of N x 365 arrays, each array contains HGT data of a day.
   */
  def readHGT(startYear: Int, endYear: Int, lat: Double, lon: Double): List[Array[Float]] = {
    val paths = (startYear to endYear).map(year => s"dat/geo/hgt.$year.nc")
    val xs = paths.map(path => readHGT(path, lat, lon))
    xs.reduce(_ ++ _)
  }

  /**
   * Extracts soil surface moisture (SSM) at a position.
   * @param path
   * @param lat
   * @param lon
   * @return an array of 365 values for a given year.
   */
  def readSSM(path: String, lat: Double, lon: Double): List[Float] = {
    val ncFile = NetcdfFiles.open(path)
    val ssm = ncFile.findVariable("soilw")
    // filter values at the station
    val (latId, lonId) = index(lat, lon)
    val data = ssm.read(s":,$latId,$lonId").reduce() // [time, lat, lon]
    ncFile.close()
    val xs = collection.mutable.ListBuffer[Float]()
    for (i <- 0 until data.getSize.toInt)
      xs += data.getFloat(i)
    xs.toList
  }

  def readSSM(startYear: Int, endYear: Int, lat: Double, lon: Double): List[Float] = {
    val paths = (startYear to endYear).map(year => s"dat/ssm/soilw.0-10cm.gauss.$year.nc")
    paths.flatMap(path => readSSM(path, lat, lon)).toList
  }

  /**
   * Extracts the wind (u-wind, v-wind) at a position (lon, lat) at some pressure levels for every day
   * in a year. The year is specified in the given path.
   *
   * @param pathU
   * @param pathV
   * @param lat
   * @param lon
   * @return a list of 366 arrays, each array contains 2 u-wind values and 2 v-wind values at level 2 (850mb) and level 9 (200mb).
   */
  def readWind(pathU: String, pathV: String, lat: Double, lon: Double): List[Array[Float]] = {
    val ncFileU = NetcdfFiles.open(pathU)
    val uwind = ncFileU.findVariable("uwnd")
    val ncFileV = NetcdfFiles.open(pathV)
    val vwind = ncFileV.findVariable("vwnd")
    val (latId, lonId) = index(lat, lon)
    val u850 = uwind.read(s":,2,$latId,$lonId").reduce()
    val u200 = uwind.read(s":,9,$latId,$lonId").reduce()
    val v850 = vwind.read(s":,2,$latId,$lonId").reduce()
    val v200 = vwind.read(s":,9,$latId,$lonId").reduce()
    // collect values
    val xs = collection.mutable.ListBuffer[Array[Float]]()
    for (i <- 0 until u850.getSize.toInt) {
      xs += Array(u850.getFloat(i), v850.getFloat(i), u200.getFloat(i), v200.getFloat(i))
    }
    ncFileU.close()
    ncFileV.close()
    xs.toList

  }

  /**
   * Reads a directory of N years of data and extract daily wind values at two pressure levels.
   *
   * @param startYear
   * @param endYear
   * @param lat
   * @param lon
   * @return a list of N x 365 arrays, each array contains u-wind, v-wind at 2 pressure levels.
   */
  def readWind(startYear: Int, endYear: Int, lat: Double, lon: Double): List[Array[Float]] = {
    val paths = (startYear to endYear).map(year => (s"dat/uvw/uwnd.$year.nc", s"dat/uvw/vwnd.$year.nc"))
    val xs = paths.map(path => readWind(path._1, path._2, lat, lon))
    xs.reduce(_ ++ _)
  }


  def main(args: Array[String]): Unit = {
    val ids = stations.foreach(p => print(index(p._1, p._2)))

    // 1. HGT
//    val xs = readHGT(1980, 2019, stations.head._1, stations.head._2)
//    println("Length of xs = " + xs.size)
//    val outputPath = Path.of("dat/geo.80-19.csv")
//    val st = xs.map(x => x.mkString(","))
//    import scala.jdk.CollectionConverters._
//    Files.write(outputPath, st.prepended("H850,H500").asJava, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)

    // 2. SLP
//    val xs = readSLP(1980, 2019, stations.head._1, stations.head._2)
//    println("Length of xs = " + xs.size)
//    val outputPath = Path.of("dat/slp.80-19.csv")
//    val st = xs.map(_.toString)
//    import scala.jdk.CollectionConverters._
//    Files.write(outputPath, st.prepended("SLP").asJava, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)

    // 3. Humidity
//    val xs = readHumidity(1980, 2019, stations.head._1, stations.head._2)
//    println("Length of xs = " + xs.size)
//    val outputPath = Path.of("dat/hum.80-19.csv")
//    import scala.jdk.CollectionConverters._
//    val st = xs.map(x => x.mkString(","))
//    Files.write(outputPath, st.prepended("HUM1000,HUM0850,HUM0500").asJava, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)

    // 4. SSM
//    val xs = readSSM(1980, 2019, stations.head._1, stations.head._2)
//    println("Length of xs = " + xs.size)
//    val outputPath = Path.of("dat/ssm.80-19.csv")
//    val st = xs.map(_.toString)
//    import scala.jdk.CollectionConverters._
//    Files.write(outputPath, st.prepended("SSM").asJava, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)

    // 5. Wind
    val xs = readWind(1980, 2019, stations.head._1, stations.head._2)
    println("Length of xs = " + xs.size)
    val outputPath = Path.of("dat/uvw.80-19.csv")
    val st = xs.map(_.mkString(","))
    import scala.jdk.CollectionConverters._
    Files.write(outputPath, st.prepended("U850,V850,U200,V200").asJava, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }
}
