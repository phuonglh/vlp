package vlp.med

case class Config(
  language: String = "en",
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y ==> there are Y/X executors
  executorMemory: String = "8g", // Z
  driverMemory: String = "8g", // D
  mode: String = "eval",
  batchSize: Int = 32,
  epochs: Int = 20,
  learningRate: Double = 5E-5
)