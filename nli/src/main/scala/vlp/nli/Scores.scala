package vlp.nli

final case class Scores(
  arch: String,
  encoder: String,
  maxSequenceLength: Int,
  embeddingSize: Int,
  encoderSize: Int,
  bidirectional: Boolean,
  tokenized: Boolean,
  trainingScores: Array[Float],
  testScore: Float
)