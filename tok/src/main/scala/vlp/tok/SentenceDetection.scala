package vlp.tok

import scala.util.matching.Regex
import scala.collection.mutable

/**
  * phuonglh
  * 
  * A sentence detection which uses simple heuristic to split a text into sentences.
  */
object SentenceDetection {
  val pattern: Regex = "([\\.?!…;:+]+|[abcdef]\\))[\\s]+[\"“]?[A-ZĐÀÁẢÃẠẤẦẨẪẬƯỪỨỬỮỰÙÚỦŨỤÔỒỐỔỖỘƠỜỚỞỠỢÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊỲÝỶỸỴ\\d]".r

  def run(text: String): Seq[String] = {
    // the character '\u00a0' is also used as a space character in many websites; replace it with the proper whitespace
    val s = text.replaceAll("\u00a0", " ").replaceAll("\\s+", " ");
    val sentences = mutable.ListBuffer[String]()
    val ms = pattern.findAllIn(s)
    var i = 0
    while (ms.hasNext) {
      val j = ms.start
      sentences.append(s.subSequence(i, j+1).toString().trim)
      i = j + 1
      ms.next()
    }
    if (i < text.size)
      sentences.append(s.subSequence(i, s.size).toString().trim)
    sentences.toSeq
  }

  def main(args: Array[String]): Unit = {
    val text = """
        Chiều 3/5, tại thông cáo nêu một số nội dung đã được xem xét, kết luận tại cuộc họp ngày 27-28/4, Uỷ ban Kiểm tra Trung ương cho biết căn cứ quy định của Đảng về xử lý kỷ luật đảng viên, Uỷ ban đề nghị Bộ Chính trị, Ban Chấp hành Trung ương xem xét, thi hành mức kỷ luật trên với ông Hiến.
        Uỷ ban Kiểm tra Trung ương xác định ông Hiến đã "vi phạm rất nghiêm trọng các quy định của pháp luật trong vụ án hình sự xảy ra tại Quân chủng Hải quân".
        Liên quan xử lý sai phạm của cán bộ Bộ Quốc phòng, Uỷ ban kiểm tra Trung ương cũng kỷ luật khai trừ ra khỏi Đảng với ông Nguyễn Văn Khuây (nguyên Phó Bí thư Đảng ủy, nguyên Sư đoàn trưởng Sư đoàn 363) và ông Vũ Duy An (nguyên Chủ nhiệm Hậu cần Sư đoàn 363). Hai ông bị xác định "vi phạm rất nghiêm trọng các quy định của pháp luật về quản lý đất đai tại Sư đoàn 363, Quân chủng Phòng không - Không quân".
    """
    val ss = run(text)
    ss.zipWithIndex.foreach(println)
  }
}