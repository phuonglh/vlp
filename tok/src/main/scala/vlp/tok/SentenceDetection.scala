package vlp.tok

import scala.util.matching.Regex
import scala.collection.mutable

/**
  * phuonglh
  * 
  * A sentence detection which uses simple heuristic to split a text into sentences.
  */
object SentenceDetection {
  val pattern: Regex = "([\\.?!…;:+-]+|[abcdef]\\))[\\s]+[\"“]?[A-ZĐÀÁẢÃẠẤẦẨẪẬƯỪỨỬỮỰÙÚỦŨỤÔỒỐỔỖỘƠỜỚỞỠỢÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊỲÝỶỸỴ\\d]".r

  def run(text: String, numberOfSentences: Int = Int.MaxValue): Seq[String] = {
    // the character '\u00a0' is also used as a space character in many websites; replace it with the proper whitespace
    val s = text.replaceAll("\u00a0", " ").replaceAll("\\s+", " ");
    val sentences = mutable.ListBuffer[String]()
    val ms = pattern.findAllIn(s)
    var i = 0
    var count = 0;
    while (ms.hasNext && count < numberOfSentences) {
      val j = ms.start
      val offset = ms.group(0).size
      sentences.append(s.subSequence(i, j + offset - 1).toString().trim)
      i = j + offset - 1
      ms.next()
      count = count + 1
    }
    if (count < numberOfSentences && i < text.size)
      sentences.append(s.subSequence(i, s.size).toString().trim)
    sentences.toSeq
  }

  def main(args: Array[String]): Unit = {
    val text = """
        Liên quan xử lý sai phạm của cán bộ Bộ Quốc phòng, Uỷ ban kiểm tra Trung ương cũng kỷ luật khai trừ ra khỏi Đảng với ông Nguyễn Văn Khuây (nguyên Phó Bí thư Đảng ủy, nguyên Sư đoàn trưởng Sư đoàn 363) và ông Vũ Duy An (nguyên Chủ nhiệm Hậu cần Sư đoàn 363). Hai ông bị xác định "vi phạm rất nghiêm trọng các quy định của pháp luật về quản lý đất đai tại Sư đoàn 363, Quân chủng Phòng không - Không quân".
        Các hoạt động khác: + Hội nghị trực tuyến <cq>Ủy ban An ninh hàng không dân dụng quốc gia</cq>. - Họp kế hoạch ứng phó biến đổi khí hậu năm 2018. - Dự buổi tiếp đoàn doanh nghiệp Thuỵ Điển cùng với <cq>Ủy ban nhân dân Thành phố</cq>. - Kiểm tra công tác chuẩn bị tổ chức Ngày Asean phòng, chống sốt xuất huyết. - Dự hoạt động giải trình tại phiên họp của <cq>Hội đồng nhân dân Thành phố</cq>. - Họp giao ban <cq>Ban quản lý Dự án Sức khoẻ dồi dào</cq>.
        a) Nội dung điều chỉnh: - Thay đổi phương án bố trí mặt bằng nhóm nhà ở thấp tầng.
        - Giảm diện tích đất ở và cây xanh để tăng diện tích đất giao thông,
        quy mô dân số không thay đổi.
        - Điều chỉnh giảm diện tích đất nhóm nhà ở cao tầng.
        - Các nội dung khác giữ nguyên theo nội dung được duyệt tại <vb>Quyết định
        số 96/QĐ-BQLKN</vb> ngày 19/10/2012.
    """
    val ss = run(text)
    ss.zipWithIndex.foreach(println)
  }
}