package vlp.vio

case class Answer(
  text: String,
  correct: Boolean
)

case class QA(
  id: String, 
  content: String, 
  answers: List[Answer]
)