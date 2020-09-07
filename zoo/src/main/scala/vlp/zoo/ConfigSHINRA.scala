package vlp.zoo

case class ConfigSHINRA(
  master: String = "local[*]",
  mode: String = "eval",
  totalCores: Int = 8,
  executorCores: Int = 8,
  executorMemory: String = "8g",
  language: String = "vi",
  dataPath: String = "/opt/data/shinra/",
  embeddingPath: String = "/opt/data/emb/",
  modelPath: String = "dat/zoo/tcl/shi/",
  numFeatures: Int = 100000,
  encoder: String = "cnn",
  embeddingDimension: Int = 300,
  encoderOutputDimension: Int = 300,
  maxSequenceLength: Int = 256,
  batchSize: Int = 64,
  epochs: Int = 50,
  learningRate: Double = 0.001,
  percentage: Double = 1.0,
  partitions: Int = 20,
  minFrequency: Int = 2,
  verbose: Boolean = false,
  inputCol: String = "body",
  classCol: String = "clazz",
  inputPath: String = "frwiki-20190121-cirrussearch-content.json"
)

