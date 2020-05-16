package vlp.idx

import java.io.IOException
import java.net.{MalformedURLException, URL}
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util.Date
import java.util.regex.Pattern

import de.l3s.boilerpipe.BoilerpipeProcessingException
import de.l3s.boilerpipe.extractors.ArticleExtractor
import org.apache.commons.io.IOUtils
import org.xml.sax.InputSource

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import org.apache.log4j.{Level, Logger}
import org.slf4j.LoggerFactory

/**
  * Indexer of financial news content for a given day from some websites.
  *
  */
object NewsIndexer {
  final val logger = LoggerFactory.getLogger(getClass.getName)
  /**
    * Extracts the main content of a news URL.
    *
    * @param url a URL to extract text.
    * @return main content in string.
    */
  def extract(url: String): String = {
    try {
      val path = new URL(url)
      val is = new InputSource
      is.setEncoding("UTF-8")
      is.setByteStream(path.openStream)
      ArticleExtractor.INSTANCE.getText(is)
    } catch {
      case e: MalformedURLException => logger.error(e.getMessage); ""
      case e: BoilerpipeProcessingException => logger.error(e.getMessage); ""
      case e: IOException => logger.error(e.getMessage); ""
    }
  }

  /**
    * Extracts URLs of a given category of a given site.
    * @param site a site
    * @param category a category in that site
    * @param regexp a regular expression to get url
    * @param filter a URL filter
    * @return a set of URLs.
    */
  private def extractURLs(site: String, category: String, regexp: String, filter: String => Boolean): Set[String] = {
    val urls = mutable.Set[String]()
    try {
      val address = site + "/" + category
      val connection = new URL(address).openConnection
      connection.setReadTimeout(5000)
      val inputStream = connection.getInputStream
      val pageContent = IOUtils.toString(inputStream, StandardCharsets.UTF_8)
      val lines = pageContent.split("\\n")
      val pattern = Pattern.compile(regexp)
      for (line <- lines) {
        val matcher = pattern.matcher(line)
        while (matcher.find()) {
          val url = matcher.group()
          if (filter(url)) urls += (site + url)
        }
      }
    } catch {
      case e: IOException => e.printStackTrace()
    }
    urls.toSet
  }

  def vnAgency(date: String): Set[String] = {
    val categories = Array("thi-truong-tai-chinh-587ct128.htm", "doanh-nghiep-doanh-nhan-145ct128.htm", "bat-dong-san-144ct128.htm", "hoi-nhap-143ct128.htm", "kinh-te-128ct0.htm")
    val urls = mutable.Set[String]()
    for (category <- categories) {
      urls ++= extractURLs("https://baotintuc.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.htm", (s: String) => s.contains(date))
    }
    logger.info("baotintuc.vn => " + urls.size)
    urls.toSet
  }

  def vtv(date: String): Set[String] = {
    val categories = Array("kinh-te.htm", "kinh-te/bat-dong-san.htm", "kinh-te/tai-chinh.htm", "kinh-te/thi-truong.htm")
    val urls = mutable.Set[String]()
    for (category <- categories) {
      urls ++= extractURLs("https://vtv.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.htm", (s: String) => s.contains(date))
    }
    logger.info("vtv.vn => " + urls.size)
    urls.toSet
  }

  def youth(date: String): Set[String] = {
    val categories = Array("tai-chinh", "doanh-nghiep", "mua-sam", "dau-tu")
    val urls = mutable.Set[String]()
    for (category <- categories) {
      urls ++= extractURLs("https://tuoitre.vn", "kinh-doanh/" + category + ".htm", "/[\\p{Alnum}/-]+(\\d{4,})\\.htm", (s: String) => s.contains(date) && !s.contains("?"))
    }
    logger.info("tuoitre.vn => " + urls.size)
    urls.toSet
  }

  def vnEconomy(date: String): Set[String] = {
    val categories = Array("tai-chinh", "chung-khoan", "dia-oc", "thi-truong", "doanh-nhan", "the-gioi", "thoi-su")
    val urls = mutable.Set[String]()
    for (category <- categories) {
      urls ++= extractURLs("http://vneconomy.vn", category + ".htm", "/[\\p{Alnum}-]+\\.htm", (s: String) => s.indexOf(date) > 0 && !s.startsWith("/news-"))
    }
    logger.info("vnEconomy.vn => " + urls.size)
    urls.toSet
  }

  def finance: Set[String] = {
    val categories = Array("nhip-song-tai-chinh-3", "thue-voi-cuoc-song-4", "chung-khoan-5", "tien-te-bao-hiem-6", "kinh-doanh-7")
    val urls = mutable.Set[String]()
    for (category <- categories) {
      urls ++= extractURLs("http://thoibaotaichinhvietnam.vn", "pages/" + category + ".aspx", "/[\\p{Alnum}/-]+\\.aspx", (s: String) => s.startsWith("/pages/"))
    }
    logger.info("thoibaotaichinhvietnam.vn => " + urls.size)
    urls.toSet
  }

  def labor(date: String): Set[String] = extractURLs("https://nld.com.vn", "/kinh-te.htm", "/[\\p{Alnum}/-]+(\\d{4,})\\.htm", (s: String) => s.contains(date) && !s.contains("?"))

  def peopleKnowledge(date: String): Set[String] = extractURLs("https://dantri.com.vn", "kinh-doanh.htm", "/[\\p{Alnum}/-]+(\\d{4,})\\.htm", (s: String) => s.contains(date) && !s.contains("?"))


  def saigonTimes: Set[String] = {
    val categories = Array("kinhdoanh", "taichinh", "doanhnghiep", "diendan", "thegioi")
    val urls = mutable.Set[String]()
    for (category <- categories) {
      urls ++= extractURLs("https://www.thesaigontimes.vn", category, "/[\\p{Alnum}/-]+\\.html", (s: String) => !s.startsWith("/SaiGonTimes"))
    }
    logger.info("thesaigontimes.vn => " + urls.size)
    urls.toSet
  }

  def vnExpress: Set[String] = {
    val categories = Array("doanh-nghiep", "bat-dong-san", "thuong-mai-dien-tu", "hang-hoa", "chung-khoan", "quoc-te")
    val urls = mutable.Set[String]()
    for (category <- categories) {
      urls ++= extractURLs("https://vnexpress.net", "kinh-doanh/" + category, "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (s: String) => s.contains(category))
    }
    logger.info("vnExpress.vn/kinh-doanh => " + urls.size)
    urls.toSet
  }

  def vnExpressInSection(section: String): Set[String] = {
    val categories = section match {
      case "kinh-doanh" => Array("doanh-nghiep", "bat-dong-san", "thuong-mai-dien-tu", "hang-hoa", "chung-khoan", "quoc-te")
      case _ => Array("")
    }
    val urls = mutable.Set[String]()
    for (category <- categories) {
      urls ++= extractURLs("https://vnexpress.net", section + "/" + category, "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (s: String) => s.contains(category))
    }
    urls.toSet
  }

  def adolescence: Set[String] = {
    extractURLs("https://thanhnien.vn/tai-chinh-kinh-doanh", "", "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (_: String) => true)
  }

  def sggp: Set[String] = {
    val categories: Array[String] = Array("kinhte/", "xahoi/")
    val urls = mutable.Set[String]()
    for (category <- categories) {
      urls ++= extractURLs("http://www.sggp.org.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (s: String) => s.contains("-"))
    }
    logger.info("sggp.org.vn => " + urls.size)
    urls.toSet
  }

  def vietnamnetInSection(section: String): Set[String] = {
    val categories = section match {
      case "kinh-doanh" => Array("tai-chinh", "dau-tu", "thi-truong", "doanh-nhan", "tu-van-tai-chinh")
      case "thoi-su" => Array("chong-tham-nhung", "quoc-hoi", "an-toan-giao-thong", "moi-truong", "quoc-phong")
      case "the-gioi" => Array("binh-luan-quoc-te", "chan-dung", "ho-so", "the-gioi-do-day", "viet-nam-va-the-gioi", "quan-su")
      case "giao-duc" => Array("nguoi-thay", "tuyen-sinh", "du-hoc", "guong-mat-tre", "goc-phu-huynh", "khoa-hoc")
      case "doi-song" => Array("gia-dinh", "song-la", "gioi-tre", "gioi-tre", "me-va-be", "du-lich", "am-thuc", "me-va-be")
      case _ => Array("")
    }
    val urls = mutable.Set[String]()
    for (category <- categories) {
      urls ++= extractURLs("https://vietnamnet.vn/vn", section + "/" + category + "/", "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (s: String) => s.contains("-"))
    }
    urls.toSet.filterNot(_.contains("/en/"))
  }

  def vietnamnetAll: Set[String] = {
    val urls = mutable.Set[String]()
    urls ++= vietnamnetInSection("kinh-doanh")
    urls ++= vietnamnetInSection("thoi-su")
    urls ++= vietnamnetInSection("the-gioi")
    urls ++= vietnamnetInSection("giao-duc")
    urls ++= vietnamnetInSection("doi-song")
    urls ++= vietnamnetInSection("phap-luat")
    urls ++= vietnamnetInSection("cong-nghe")
    urls ++= vietnamnetInSection("suc-khoe")
    urls ++= vietnamnetInSection("bat-dong-san")
    logger.info("vietnamnet.vn => " + urls.size)
    urls.toSet
  }

  def pioneer: Set[String] = {
    val categories = Array("kinh-te/", "kinh-te-thi-truong/", "kinh-te-doanh-nghiep/", "kinh-te-doanh-nhan/", "kinh-te-chung-khoan")
    val urls = mutable.Set[String]()
    for (category <- categories) {
      urls ++= extractURLs("https://www.tienphong.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.tpo", (s: String) => s.contains("kinh-te"))
    }
    logger.info("tienphong.vn => " + urls.size)
    urls.toSet
  }

  def customs: Set[String] = {
    val categories = Array("hai-quan", "tai-chinh", "kinh-te", "doanh-nghiep")
    val urls = mutable.Set[String]()
    for (category <- categories)
      urls ++= extractURLs("https://haiquanonline.com.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.html", _.contains("-"))
    logger.info("haiquanonline.com.vn => " + urls.size)
    urls.toSet
  }

  def vietnamFinance: Set[String] = {
    val categories = Array("tieu-diem.htm", "tai-chinh.htm", "ngan-hang.htm", "thi-truong.htm", "do-thi.htm", "tai-chinh-quoc-te",
      "ma.htm", "startup.htm", "nhan-vat.htm", "thue.htm", "tai-chinh-tieu-dung.htm", "dien-dan-vnf.htm")
    val urls = mutable.Set[String]()
    for (category <- categories)
      urls ++= extractURLs("https://vietnamfinance.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.htm", _.contains("-"))
    logger.info("vietnamfinance.vn => " + urls.size)
    urls.toSet
  }

  def fastStockNews: Set[String] = {
    val categories = Array("chung-khoan/", "thuong-truong/", "bao-hiem/", "doanh-nghiep/", "tien-te/")
    val urls = mutable.Set[String]()
    for (category <- categories)
      urls ++= extractURLs("https://tinnhanhchungkhoan.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.html",
        s => s.contains("-") && !s.contains("/toa-soan/") && !s.contains("/dai-hoi-co-dong/") && !s.contains("/don-doc/") && !s.contains("/cuoc-song/"))
    logger.info("tinnhanhchungkhoan.vn => " + urls.size)
    urls.toSet
  }

  def light: Set[String] = {
    val categories = Array("tin-tuc", "dien-dan", "quan-ly-khoa-hoc", "khoa-hoc-cong-nghe", "doi-moi-sang-tao", "giao-duc", "van-hoa")
    val urls = mutable.Set[String]()
    for (category <- categories)
      urls ++= extractURLs("http://tiasang.com.vn", category, "/[\\p{Alnum}/-]+(\\d{5,})", _.contains("-"))
    logger.info("tiasang.com.vn => " + urls.size)
    urls.toSet
  }

  def vcci: Set[String] = {
    val categories = Array("cate/693/vcci.html", "cate/702/doi-ngoai.html", "cate/728/kinh-te-thi-truong.html", "cate/714/xuat-nhap-khau.html",
      "cate/721/dau-tu.html", "cate/707/tai-chinh.html", "cate/734/doanh-nghiep.html", "cate/740/bat-dong-san.html", "cate/746/cong-nghe.html")
    val urls = mutable.Set[String]()
    for (category <- categories)
      urls ++= extractURLs("http://www.vccinews.vn", category, "/news/\\d{5,}/[\\p{Alnum}/-]+\\.html", _.contains("-"))
    logger.info("vccinews.vn => " + urls.size)
    urls.toSet
  }

  def vnExpressOthers: Set[String] = {
    val urls = new ListBuffer[String]
    urls ++= vnExpressInSection("thoi-su")
    urls ++= vnExpressInSection("goc-nhin")
    urls ++= vnExpressInSection("the-gioi")
    urls ++= vnExpressInSection("giai-tri")
    urls ++= vnExpressInSection("phap-luat")
    urls ++= vnExpressInSection("giao-duc")
    urls ++= vnExpressInSection("suc-khoe")
    urls ++= vnExpressInSection("doi-song")
    urls ++= vnExpressInSection("du-lich")
    urls ++= vnExpressInSection("khoa-hoc")
    urls ++= vnExpressInSection("so-hoa")
    urls ++= vnExpressInSection("oto-xe-may")
    logger.info("vnExpress.net/Others => " + urls.size)
    urls.toSet
  }

  def vir: Set[String] = {
    val urls = mutable.Set[String]()
    urls ++= extractURLs("https://baodautu.vn", "", "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (_: String) => true)
    urls ++= extractURLs("https://dautubds.baodautu.vn", "", "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (_: String) => true)
    val categories = Set("thoi-su-d1", "dau-tu-d2", "quoc-te-d54", "doanh-nghiep-d3", "doanh-nhan-d4", "ngan-hang-d5", "tai-chinh-chung-khoan-d6")
    for (category <- categories) {
      urls ++= extractURLs("https://baodautu.vn/" + category, "", "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (s: String) => true)
    }
    logger.info("baodautu.vn => " + urls.size)
    urls.toSet
  }

  def transport: Set[String] = {
    val urls = mutable.Set[String]()
    val categories = Seq("kinh-te", "thi-truong", "tai-chinh", "bao-hiem", "bat-dong-san")
    for (category <- categories)
      urls ++= extractURLs("https://www.baogiaothong.vn", category, "/[\\p{Alnum}/-]+(\\d{4,})\\.html", (_: String) => true)
    logger.info("baogiaothong.vn => " + urls.size)
    urls.toSet
  }

  def run(date: String): Unit = {
    System.setProperty("http.agent", "Chrome")
    System.setProperty("https.protocols", "TLSv1,TLSv1.1,TLSv1.2")

    import scala.collection.JavaConversions._
    val existingURLs = MySQL.getURLs
    logger.info(s"#(existingURLs) = ${existingURLs.size}")
    val urls = mutable.Set[String]()
    urls ++= vnAgency(date)
    urls ++= vtv(date)
    urls ++= youth(date)
    urls ++= vnEconomy(date)
    urls ++= labor(date)
    urls ++= saigonTimes
    urls ++= vnExpress
    urls ++= adolescence
    urls ++= sggp
    urls ++= pioneer
    urls ++= customs
    urls ++= vietnamFinance
    urls ++= fastStockNews
    urls ++= light
    urls ++= vcci
    urls ++= vnExpressOthers
    urls ++= vietnamnetAll
    urls ++= vir
    urls ++= finance
    urls ++= transport

    logger.info(s"#(totalURLs) = ${urls.size}")
    val novelUrls = urls.diff(existingURLs)
    logger.info(s"#(novelURLs) = ${novelUrls.size}")
    val news = novelUrls.par.map(url => {
      logger.info(url)
      val content = extract(url)
      new News(url, content, new Date())
    }).toList
    if (news.nonEmpty) {
      val accept = (s: String) => (s.size >= 500 && !s.contains("<div") && !s.contains("<table") && !s.contains("</p>"))
      import scala.collection.JavaConverters._
      val xs = news.filter(x => accept(x.getContent)).asJava
      // update the ES index
      Indexer.indexManyNews(xs)
      logger.info(s"#(indexedNews) = ${xs.size}")
      // update the MySQL database `url`
      MySQL.insert(novelUrls)
      logger.info(s"#(insertedURLs) = ${novelUrls.size}")
    }    
    Thread.sleep(1000)
    Indexer.close()
  }

  def test: Unit = {
    val urls = transport
    urls.foreach(println)
    println(s"#(urls) = ${urls.size}")
  }

  def main(args: Array[String]): Unit = {
    test
    if (args.size >= 1) {
      run(args(0))
    } else {
      val dateFormat = new SimpleDateFormat("yyyyMMdd")
      val currentDate = dateFormat.format(new Date())
      run(currentDate)
    }
  }


}
