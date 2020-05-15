package vlp.vdr

import scala.collection.mutable

/**
  * phuonglh, 10/31/17, 20:06
  */
object VieMap {
  final val diacritics = Map[Char, Char](
    'a' -> 'a', 'à' -> 'a', 'á' -> 'a', 'ả' -> 'a', 'ã' -> 'a', 'ạ' -> 'a',
    'â' -> 'a', 'ầ' -> 'a', 'ấ' -> 'a', 'ẩ' -> 'a', 'ẫ' -> 'a', 'ậ' -> 'a',
    'ă' -> 'a', 'ằ' -> 'a', 'ắ' -> 'a', 'ẳ' -> 'a', 'ẵ' -> 'a', 'ặ' -> 'a',
    'e' -> 'e', 'è' -> 'e', 'é' -> 'e', 'ẻ' -> 'e', 'ẽ' -> 'e', 'ẹ' -> 'e',
    'ê' -> 'e', 'ề' -> 'e', 'ế' -> 'e', 'ể' -> 'e', 'ễ' -> 'e', 'ệ' -> 'e',
    'o' -> 'o', 'ò' -> 'o', 'ó' -> 'o', 'ỏ' -> 'o', 'õ' -> 'o', 'ọ' -> 'o',
    'i' -> 'i', 'ì' -> 'i', 'í' -> 'i', 'ỉ' -> 'i', 'ĩ' -> 'i', 'ị' -> 'i',
    'ô' -> 'o', 'ồ' -> 'o', 'ố' -> 'o', 'ổ' -> 'o', 'ỗ' -> 'o', 'ộ' -> 'o',
    'ơ' -> 'o', 'ờ' -> 'o', 'ớ' -> 'o', 'ở' -> 'o', 'ỡ' -> 'o', 'ợ' -> 'o',
    'u' -> 'u', 'ù' -> 'u', 'ú' -> 'u', 'ủ' -> 'u', 'ũ' -> 'u', 'ụ' -> 'u',
    'ư' -> 'u', 'ừ' -> 'u', 'ứ' -> 'u', 'ử' -> 'u', 'ữ' -> 'u', 'ự' -> 'u',
    'y' -> 'y', 'ỳ' -> 'y', 'ý' -> 'y', 'ỷ' -> 'y', 'ỹ' -> 'y', 'ỵ' -> 'y',
    'd' -> 'd', 'đ' -> 'd'
  )
  final val singles = Map[String, List[String]](
    "a" -> List("a", "à", "á", "ả", "ã", "ạ", "â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "ă", "ằ", "ắ", "ẳ", "ẵ", "ặ"),
    "e" -> List("e", "è", "é", "ẻ", "ẽ", "ẹ", "ê", "ề", "ế", "ể", "ễ", "ệ"),
    "o" -> List("o", "ò", "ó", "ỏ", "õ", "ọ", "ô", "ồ", "ố", "ổ", "ỗ", "ộ", "ơ", "ờ", "ớ", "ở", "ỡ", "ợ"),
    "i" -> List("i", "ì", "í", "ỉ", "ĩ", "ị"),
    "u" -> List("u", "ù", "ú", "ủ", "ũ", "ụ", "ư", "ừ", "ứ", "ử", "ữ", "ự"),
    "y" -> List("y", "ỳ", "ý", "ỷ", "ỹ", "ỵ"),
    "d" -> List("d", "đ")
  )

  final val pairs = Map[String, List[String]](
    "ao" -> List("ao", "ào", "áo", "ảo", "ão", "ạo"),
    "ai" -> List("ai", "ài", "ái", "ải", "ãi", "ại"),
    "au" -> List("au", "àu", "áu", "ảu", "ãu", "ạu", "âu", "ầu", "ấu", "ẩu", "ẫu", "ậu"),
    "ay" -> List("ay", "ày", "áy", "ảy", "ãy", "ạy", "ây", "ầy", "ấy", "ẩy", "ẫy", "ậy"),
    "eo" -> List("eo", "èo", "éo", "ẻo", "ẽo", "ẹo"),
    "eu" -> List("êu", "ều", "ếu", "ểu", "ễu", "ệu"),
    "oa" -> List("oa", "oà", "oá", "oả", "oã", "oạ", "oă", "oằ", "oắ", "oẳ", "oẵ", "oặ"),
    "oe" -> List("oe", "oè", "oé", "oẻ", "oẽ", "oẹ"),
    "oi" -> List("oi", "òi", "ói", "ỏi", "õi", "ọi", "ơi", "ời", "ới", "ởi", "ỡi", "ợi", "ôi", "ồi", "ối", "ổi", "ỗi", "ội"),
    "oo" -> List("oo", "oò", "oó", "oỏ", "oõ", "oọ"),
    "ia" -> List("ia", "ìa", "ía", "ỉa", "ĩa", "ịa"),
    "ie" -> List("iê", "iề", "iế", "iể", "iễ", "iệ"),
    "io" -> List("io", "iò", "ió", "iỏ", "iõ", "iọ", "iô", "iồ", "iố", "iổ", "iỗ", "iộ", "iơ", "iờ", "iớ", "iở", "iỡ", "iợ"),
    "iu" -> List("iu", "ìu", "íu", "ỉu", "ĩu", "ịu"),
    "ua" -> List("ua", "ùa", "úa", "ủa", "ũa", "ụa", "ưa", "ừa", "ứa", "ửa", "ữa", "ựa", "uă", "uằ", "uắ", "uẳ", "uẵ", "uặ", "uâ", "uầ", "uấ", "uẩ", "uẫ", "uậ"),
    "ue" -> List("uê", "uề", "uế", "uể", "uễ", "uệ"),
    "uo" -> List("uô", "uồ", "uố", "uổ", "uỗ", "uộ", "uơ", "uờ", "uớ", "uở", "uỡ", "uợ", "ươ", "ườ", "ướ", "ưở", "ưỡ", "ượ"),
    "ui" -> List("ui", "ùi", "úi", "ủi", "ũi", "ụi", "ưi", "ừi", "ứi", "ửi", "ữi", "ựi"),
    "uu" -> List("ưu", "ừu", "ứu", "ửu", "ữu", "ựu"),
    "uy" -> List("uy", "uỳ", "uý", "uỷ", "uỹ", "uỵ"),
    "ye" -> List("yê", "yề", "yế", "yể", "yễ", "yệ")
  )

  final val triples = Map[String, List[String]](
    "ieu" -> List("iêu", "iều", "iếu", "iểu", "iễu", "iệu"),
    "yeu" -> List("yêu", "yều", "yếu", "yểu", "yễu", "yệu"),
    "oai" -> List("oai", "oài", "oái", "oải", "oãi", "oại"),
    "oao" -> List("oao", "oào", "oáo", "oảo", "oão", "oạo"),
    "oay" -> List("oay", "oày", "oáy", "oảy", "oãy", "oạy"),
    "oeo" -> List("oeo", "oèo", "oéo", "oẻo", "oẽo", "oẹo"),
    "uao" -> List("uao", "uào", "uáo", "uảo", "uão", "uạo"),
    "uay" -> List("uây", "uầy", "uấy", "uẩy", "uẫy", "uậy"),
    "uoi" -> List("uôi", "uồi", "uối", "uổi", "uỗi", "uội", "ươi", "ười", "ưới", "ưởi", "ưỡi", "ượi"),
    "uou" -> List("ươu", "ườu", "ướu", "ưởu", "ưỡu", "ượu"),
    "uya" -> List("uya"),
    "uyu" -> List("uyu", "uỳu", "uýu", "uỷu", "uỹu", "uỵu"),
    "uye" -> List("uyê", "uyề", "uyế", "uyể", "uyễ", "uyệ")
  )

  def threeMaps: (mutable.Map[String, String], mutable.Map[String, String], mutable.Map[String, String]) = {
    val singles = mutable.Map[String, String]()
    VieMap.singles.keySet.foreach(e => {
      VieMap.singles(e).foreach(v => singles.put(v, e))
    })
    val pairs = mutable.Map[String, String]()
    VieMap.pairs.keySet.foreach(e => {
      VieMap.pairs(e).foreach(v => pairs.put(v, e))
      pairs.put(e, e)
    })
    val triples = mutable.Map[String, String]()
    VieMap.triples.keySet.foreach(e => {
      VieMap.triples(e).foreach(v => triples.put(v, e))
      triples.put(e, e)
    })
    (singles, pairs, triples)
  }
  
  def vowels: mutable.Map[String, String] = {
    val vowels = mutable.Map[String, String]()
    vowels.put("òa", "oà")
    vowels.put("óa", "oá")
    vowels.put("ỏa", "oả")
    vowels.put("õa", "oã")
    vowels.put("ọa", "oạ")
    vowels.put("òe", "oè")
    vowels.put("óe", "oé")
    vowels.put("ỏe", "oẻ")
    vowels.put("õe", "oẽ")
    vowels.put("ọe", "oẹ")
    vowels.put("ùy", "uỳ")
    vowels.put("úy", "uý")
    vowels.put("ủy", "uỷ")
    vowels.put("ũy", "uỹ")
    vowels.put("ụy", "uỵ")
    vowels
  }
  
  def normalize(text: String): String = {
    val vs = vowels
    var result = text
    for (k <- vs.keySet)
      if (result.contains(k))
        result = result.replaceAll(k, vs(k))
    result
  }
}
