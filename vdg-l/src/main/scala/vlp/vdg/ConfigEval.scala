package vlp.vdg

case class ConfigEval(
  task: String = "vdg",
  dataPath: String = "",
  dataRatio: Double = 1.0,
  modelPath: String = "",
  modelType: Int = 0,
  recurrentType: String = "gru",
  numLayers: Int = 1,
  hiddenUnits: Int = 50,
  encoderOutputSize: Int = 128, // for use with Transformers
  trainingScore: Double = 0,
  validationScore: Double = 0,
  testScore: Double = 0,
  trainingTime: Long = 0
)
