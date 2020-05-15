package vlp.vdr

import scala.io.Source

/**
  * phuonglh, 10/12/17, 3:15 PM
  */
object IO {
  /**
    * Reads  a list of sentences from a path.
    * @param path
    * @param resource
    * @return a list of sentences
    */
  def readSentences(path: String, resource: Boolean = false): List[String] = {
    if (resource) {
      Source.fromInputStream(getClass.getResourceAsStream(path), "UTF-8")
        .getLines().filterNot(_.trim.isEmpty).toList
        .map(x => VieMap.normalize(x))
    } else {
      Source.fromFile(path, "UTF-8").getLines().filterNot(_.trim.isEmpty).toList
    }
  }

  /**
    * Read categories of Vin-Ecommerce data.
    * @param path
    * @param resource
    * @param lowerCase
    * @return a list of strings
    */
  def readCategories(path: String, resource: Boolean = false, lowerCase: Boolean = true): List[String] = {
    val data = if (resource) {
      Source.fromInputStream(getClass.getResourceAsStream(path), "UTF-8")
    } else {
      Source.fromFile(path, "UTF-8")
    }
    val list = data.getLines().filterNot(_.trim.isEmpty).toList
      .map(e => e.replaceAll("[–,&-]+", ""))
      .map(e => e.replaceAll("[/]+", " "))
      .map(e => e.replaceAll("\\s+", " "))
      .map(x => VieMap.normalize(x))
    if (lowerCase) list.map(_.toLowerCase) else list
  }
  
  def readMappings(path: String): Map[String, String] = {
    val data = Source.fromFile(path, "UTF-8")
    val list = data.getLines().filterNot(_.trim.isEmpty).toList
      .map(line => {
        val uv = line.split(";")
        if (uv.size > 1) (uv(0).trim, uv(1).trim); else ("NA", "NA")
      }).filterNot(_._1 == "NA")
    list.toMap
  }
}
