package vlp.ner

import scala.collection.mutable.ListBuffer
import vlp.tok.WordShape

/**
 * @author Phuong LE-HONG, phuonglh@gmail.com
 * <p>
 * Oct 5, 2016, 1:17:45 PM
 * <p>
 *	Token regular expression.
 */

object WordRegexp {
  
  type WordSet = String => Boolean 
  
  def t(sentence: Sentence, shapes: List[String]): List[Int] = {
    val words = sentence.tokens.map(_.word).toList
    t(words, shapes)
  }
  
  /**
   * Matches a list of shape names to a sequence of words and returns 
   * a list of positions which are matched. The shape names are supposed 
   * to be provided by [[WordShape]].
   * @param words a word sequence
   * @param shapes a shape name sequence
   * @return a list of start positions which are matched.
   */
  def t(words: List[String], shapes: List[String]): List[Int] = {
    def matches(words: List[String]): Boolean = {
      val pairs = words.map(w => WordShape.shape(w)) zip shapes
      pairs.forall(p => p._1 == p._2)
    }
    
    val k = shapes.length
    val n = words.length
    if (k > n) List[Int]() else {
      val ids = ListBuffer[Int]()
      for (i <- 0 until n-k+1 if (matches(words.slice(i, i+k)))) 
        ids += i
      ids.toList
    }
  }
  
  def f(words: List[String], pattern: List[WordSet]): List[Int] = {
    def matches(words: List[String]): Boolean = {
      val pairs = words zip pattern
      pairs.forall(p => p._2(p._1))
    }
    
    val k = pattern.length
    val n = words.length
    if (k > n) List[Int]() else {
      val ids = ListBuffer[Int]()
      for (i <- 0 until n-k+1 if (matches(words.slice(i, i+k))))
        ids += i
      ids.toList
    }
  }

  def f(sentence: Sentence, pattern: List[WordSet]): List[Int] = {
    val words = sentence.tokens.map(_.word).toList
    f(words, pattern)
  }
  
  /**
   * Annotates regexp type for each token of a sentence. 
   * @param sentence
   * @return that sentence with regular expression types added. 
   */
  def annotate(sentence: Sentence): Sentence = {
    // recursive function to update the sentence
    def annotate(startIndex: Int, endIndex: Int): Sentence = {
      val x = ListBuffer[(Int, Int, String)]()
      for (name <- regexps.keys) {
        val pattern = regexps(name)
        val positions = f(sentence.slice(startIndex, endIndex), pattern)
        if (!positions.isEmpty) 
          positions.foreach(i => x.append((startIndex + i,  pattern.length, name)))
      }
      if (!x.isEmpty) {
        val m = x.maxBy(_._2)
        for (j <- 0 until m._2) {
          val t = sentence.tokens(m._1 + j)
          val kv = (Label.RegexpType, m._3)
          sentence.tokens.update(m._1 + j, Token(t.word, t.annotation + kv))
        }
        // recursively update left and right parts of the sentence
        if (m._1 > startIndex) {
          annotate(startIndex, m._1)
        }
        if (m._1 + m._2 < endIndex) {
          annotate(m._1 + m._2, endIndex)
        }
      }
      sentence
    }
    annotate(0, sentence.length)
  }

  
  val fNumber  = (w: String) => { WordShape.shape(w) == "number" }
  
  val fCapital = (w: String) => { WordShape.shape(w) == "capitalized" }
  
  val fAllcaps  = (w: String) => { WordShape.shape(w) == "allcaps" }
  
  val fName   = (w: String) => { WordShape.shape(w) == "name" || WordShape.shape(w) == "allcaps" }
  
  val fCode = (w: String) => { WordShape.shape(w) == "code" }
  
  val fProvince = (word: String) => {
    val w = word.toLowerCase
    w == "tỉnh" || w == "thành_phố" || w == "tp." || w == "tp" || 
    w == "huyện" || w == "quận" || w == "xã" || w == "phường" || 
    w == "thị_trấn" || w == "thôn" || w == "bản" || w == "làng" || w == "xóm" || w == "ấp" 
  }
  
  val fPress = (word: String) => {
    val w = word.toLowerCase
    w == "báo" || w == "tờ" || w == "tạp_chí" || w == "đài" || w == "thông_tấn_xã" 
  }
  
  val fCommunist = (word: String) => {
    val w = word.toLowerCase
    w == "thành_ủy" || w == "tỉnh_ủy" || w == "quận_ủy" || w == "huyện_ủy" || w == "xã_ủy" || w == "đảng_ủy"  
  }
  
  val fPolice = (word: String) => {
    val w = word.toLowerCase
    w == "công_an" | w == "cảnh_sát" 
  }
  
  val fSchool = (w: String) => {
    w == "ĐH" || w == "CĐ" || w == "THPT" || w == "THCS" || w == "tiểu_học"
  }
  
  val fInstitution = (word: String) => {
    val w = word.toLowerCase
    w == "trường" || w ==  "học_viện" | w == "viện" || w == "institute" || w == "university" 
  }
  
  val fCompany = (word: String) => {
    val w = word.toLowerCase
    w == "công_ty" || w == "công_ty_cổ_phần" || w == "tập_đoàn" || w == "hãng" || w == "xí_nghiệp"
  }
  
  val fUnion = (word: String) => {
    val w = word.toLowerCase
    w == "liên_hiệp" || w == "hội" || w == "hợp_tác_xã" || w == "câu_lạc_bộ" || 
      w == "trung_tâm" || w == "liên_đoàn" || w == "tổng_liên_đoàn"
  }
  
  val fMilitary = (word: String) => {
    val w = word.toLowerCase
    w == "sư_đoàn" || w == "lữ_đoàn" || w == "trung_đoàn" || w == "tiểu_đoàn" || 
      w == "quân_khu" || w == "liên_khu" 
  }
  
  val fMinistryPrefix = (word: String) => {
    val w = word.toLowerCase
    w == "bộ" || w == "ủy_ban" 
  }
  
  val fMinistry = (word: String) => {
    val w = word.toLowerCase
    w == "chính_trị" || w == "ngoại_giao" || w == "quốc_phòng" || w == "công_an" || 
      w == "tư_pháp" || w == "tài_chính" || w == "công_thương" || w == "xây_dựng" ||
      w == "nội_vụ" || w == "y_tế" || w == "ngoại_giao" || w == "lao_động" || 
      w == "giao_thông" || w == "thông_tin" || w == "tt" || w == "giáo_dục" || w == "gd" ||
      w == "nông_nghiệp" || w == "nn" || w == "kế_hoạch" || w == "kh" ||   
      w == "khoa_học" || w == "kh" || w == "văn_hóa" || w == "tài_nguyên" || w == "tn" || 
      w == "dân_tộc"
  }
  
  val fDepartmentPrefix = (word: String) => {
    val w = word.toLowerCase
    w ==  "sở" || w == "phòng" || w == "ban" || w == "chi_cục" || w == "tổng_cục" 
  }
  
  val fVillage = (word: String) => {
    val w = word.toLowerCase
    w == "quận" || w == "q" || w == "q." || w == "ấp" || w == "quán" || w == "khu" ||
      w == "tổ" || w == "khóm" || w == "xóm" || w == "trạm" || w == "số" || w == "ngách" || w == "ngõ"
  }
  
  val fRegion = (word: String) => {
    val w = word.toLowerCase
    w == "bang" || w == "nước" || w == "vùng" || w == "miền"
  }

  val fLocPrefix = (word: String) => {
    val w = word.toLowerCase
    w == "sông" || w == "núi" || w == "chợ" || w == "châu" || 
    w == "đảo" || w == "đèo" || w == "cầu" || w == "đồi" || w == "đồn" || 
    w == "thủ_đô" || w == "khách_sạn" || w == "sân_bay" || w == "nhà_hàng" || w == "cảng" ||
    w == "đường" || w == "phố" || w == "đại_lộ" || w == "chung_cư" || w == "rạch" ||
    w == "hồ" || w == "kênh"  
  }
  
  val fRoad = (word: String) => {
    val w = word.toLowerCase
    w == "tỉnh_lộ" || w == "quốc_lộ"  
  }
  
  val fParty = (word: String) => {
    word == "Đảng" || word == "đảng" 
  }
  
  val regexps = Map(
    // ORG
    "orgAdmin" -> List(fAllcaps, fProvince, fName), 
    "orgPress" -> List(fPress, fName),
    "orgCommunist1" -> List(fCommunist, fName),
    "orgCommunist2" -> List(fCommunist, fProvince, fName),
    "orgPolice" -> List(fPolice, fProvince, fName), 
    "orgSchool1" -> List(fSchool, fName), 
    "orgSchool2" -> List(fInstitution, fSchool, fName),
    "orgSchool3" -> List(fName, fInstitution), 
    "orgCompany" -> List(fCompany, fName),
    "orgUnion" -> List(fUnion, fName),
    "orgMilitary" -> List(fMilitary, fNumber),
    "orgMinistry" -> List(fMinistryPrefix, fMinistry), 
    "orgDepartment1" -> List(fDepartmentPrefix, fMinistry),
    "orgDepartment2" -> List(fDepartmentPrefix, fMinistry, fProvince, fName),
    "orgParty1" -> List(fParty, fName), 
    "orgParty2" -> List(fParty, fCapital, fName), 
    // LOC
    "locProvince" -> List(fProvince, fName), 
    "locVillage" -> List(fVillage, fNumber),
    "locRegion" -> List(fRegion, fName),
    "locGeneral" -> List(fLocPrefix, fName),
    "locRoad1" -> List(fRoad, fCode),
    "locRoad2" -> List(fRoad, fNumber),
    "locAddress" -> List(fNumber, fName)
  )

  def main(args: Array[String]): Unit = {
    var words = ListBuffer("UNBD", "tỉnh", "Cà_Mau", "đã", "xây", 
        "TP", "tiểu_học", "Hữu_Nghị", "báo", "Tuổi_Trẻ", "đưa", 
        "HĐND", "thành_phố", "Hà_Nội", "gặp", "UBND", "TP.", "HCM")
    var sentence = Sentence(words.map(w => Token(w, Map[Label.Value, String]())))
    println(sentence)
    println(annotate(sentence))
    println(sentence)
    
    println()
    
    words = ListBuffer("hôm_qua", "tỉnh_ủy", "Hà_Tĩnh", "đã", "họp_báo", "Đảng_ủy", "huyện", "Chợ_Lách", "gặp", "công_an", "tp", "Buôn_Mê_Thuột")
    sentence = Sentence(words.map(w => Token(w, Map[Label.Value, String]())))
    println(sentence)
    println(annotate(sentence))
    println(sentence)
    
    words = ListBuffer("tỉnh", "Thừa_Thiên", "18", "Hàng_Bồ", "ấp", "18", "tỉnh_lộ", "20A", "quốc_lộ", "18")
    sentence = Sentence(words.map(w => Token(w, Map[Label.Value, String]())))
    println(sentence)
    println(annotate(sentence))
    println(sentence)
    
    words = ListBuffer("đảng", "Cộng_sản", "VN", "Đảng", "Xanh", "Pháp", "đảng", "Dân_chủ")
    sentence = Sentence(words.map(w => Token(w, Map[Label.Value, String]())))
    println(sentence)
    println(annotate(sentence))
    println(sentence)
    
  }
}