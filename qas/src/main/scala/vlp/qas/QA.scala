package vlp.qas

final case class QA(
  id: Option[Any],
  question: String,
  questionDetail: String,
  answer: String, 
  questionKeywords: Option[Any],
  questionType: String,
  area: String,
  source: String
)


final case class Q(id: String, question: String, questionDetail: String, keywords: List[String])