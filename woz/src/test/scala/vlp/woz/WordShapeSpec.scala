package vlp.woz

import org.scalatest.flatspec.AnyFlatSpec

/**
  * phuonglh@gmail.com
  *
  */
class WordShapeSpec extends AnyFlatSpec {

  "WordShape" should "detect numbers" in {
    val tokens = Array("1234", "77.88", "-80.25", "+1.27")
    for (t <- tokens)
      assert(WordShape.shape(t) === "number")
  }

  it should "detect urls" in {
    val tokens = Array("http://vnu.edu.vn", "https://vnexpress.net/", 
      "https://tuoitre.vn/de-nghi-khong-tieu-thu-trung-gia-cam-bay-ban-duoi-danh-nghia-giai-cuu-20230227162830257.htm",
      "https://cafef.vn/thi-truong-giam-manh-khoi-ngoai-manh-tay-ban-rong-660-ty-dong-trong-phien-dau-tuan-20230227155111378.chn"
    )
    for (t <- tokens)
      assert(WordShape.shape(t) === "url")
  }

  it should "detect emails" in {
    val tokens = Array("phuonglh@gmail.com", "khanhlh2207@gmail.com")
    for (t <- tokens)
      assert(WordShape.shape(t) === "email")
  }

  it should "detect allcaps" in {
    val tokens = Array("USA", "HN", "TPHCM", "UBND")
    for (t <- tokens)
      assert(WordShape.shape(t) === "allcaps")
  }

  it should "detect number sequences" in {
    val tokens = Array("0969 0014 80", "+33 71 71 80 12")
    for (t <- tokens)
      assert(WordShape.shape(t) === "numSeq")
  }

  it should "detect percentages" in {
    val tokens = Array("86%", "-13.5%", "+21%", "+21.156%", "-17,23%")
    for (t <- tokens)
      assert(WordShape.shape(t) === "percentage")
  }

  it should "detect times" in {
    val tokens = Array("3:30", "18:01", "17:15", "24:00")
    for (t <- tokens)
      assert(WordShape.shape(t) === "time")
  }
  
  it should "detect dates" in {
    val tokens = Array("20/10/1980", "22/07/12", "12/07", "07/12", "02/1985")
    for (t <- tokens)
      assert(WordShape.shape(t) === "date")
  }

  it should "detect punctuations" in {
    val tokens = Array("?", ".", ",", ";", ":", """"""", """'""")
    for (t <- tokens)
      assert(WordShape.shape(t) === "punctuation")
  }

  it should "detect empty word shape" in {
    assert(WordShape.shape("abcd") === "")
  }
}
