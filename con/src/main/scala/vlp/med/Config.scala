package vlp.med

case class Config(
  language: String = "en",
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y ==> there are Y/X executors
  executorMemory: String = "8g", // Z
  driverMemory: String = "8g", // D
  mode: String = "eval",
  modelType: String = "u",
  batchSize: Int = 32,
  fraction: Double = 1.0f,
  threshold: Double = 0.1,
  epochs: Int = 20,
  learningRate: Double = 1E-4
)