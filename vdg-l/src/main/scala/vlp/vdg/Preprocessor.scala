package vlp.vdg

import scala.io.Source
import java.nio.file.Files
import java.io.File
import java.io.BufferedWriter
import java.io.FileWriter
import java.nio.charset.StandardCharsets

/**
  * phuonglh, for FTEL address project.
  * November 30, 2022
  * 
  */
object Preprocessor {
  val map = Map(
    """P\.(?!\d)""" -> "Phường ", // ambiguity with P.1, P.2 (ward of HCMC)
    """Q\.?""" -> "Quận ",
    """tp\.""" ->	"Thành phố", 
    """T[Pp]\.""" -> "Thành phố ",
    """T\.[Tt]rệt""" ->	"Tầng Trệt",
    """H\.""" -> "Huyện ",
    "TT" -> "Thị trấn",
    """T\.[Tt]rấn""" ->	"Thị trấn",
    """TX\.""" -> "Thị xã ",
    "TPHCM" -> "Thành phố Hồ Chí Minh", 
    "HCM" -> "Hồ Chí Minh", 
    "btân" ->	"Bình Tân",
    "tp," ->	"tân phú,",
    "tb," ->	"tân bình,",
    "bt,"	-> "bình thạnh,",
    "hm,"	-> "hóc môn,",
    "btân," ->	"bình tân,",
    "qpn"	-> "quận phú phuận",
    "(pn|PN)," -> "phú nhuận,",
    "(nb|NB)," -> "nhà bè,", 
    "(bc|BC)," ->	"bình chánh,",
    "(gv|GV)," -> "gò vấp,",
    """\(.+\)""" -> "", // remove everything inside parentheses    
  )

  def replace(input: String): String = {
    var s = input
    for (u <- map.keySet) {
      s = s.replaceAll(u, map.get(u).get)
    }
    return s.replaceAll("""\s+""", " ").replaceAll(" ,", ",")
  }

  def main(args: Array[String]): Unit = {
    if (args.isEmpty) {
      System.out.println("Need an input text file!")
      return
    }
    val inp = args(0)
    val out = if (args.size > 1) args(1) else ""
    val xs = Source.fromFile(inp, "UTF-8").getLines().toList
    val ys = xs.par.map(x => replace(x)).toList
    if (out.nonEmpty) {
      val file = new File(out)
      val bw = new BufferedWriter(new FileWriter(file, StandardCharsets.UTF_8))
      for (y <- ys) {
        bw.write(y)
        bw.write("\n")
      }
      bw.close()
    } else {
      ys.foreach(println)
    }
  }
}
