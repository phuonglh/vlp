package vlp.nli

final case class Scores(
  arch: String,
  encoder: String,
  maxSequenceLength: Int,
  embeddingSize: Int,
  encoderSize: Int,
  trainingScores: Array[Float],
  testScore: Double
)