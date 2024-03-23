package vlp.med

case class Config(
  language: String = "en",
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y
  executorMemory: String = "8g", // Z
  driverMemory: String = "8g", // D
  mode: String = "eval",
  modelType: String = "u",
  batchSize: Int = 64,
  fraction: Double = 1.0f,
  threshold: Double = 0.1,
  epochs: Int = 100,
  learningRate: Double = 5E-4,
  sparseLabel: Boolean = false,
  hiddenSize: Int = 256,
  concat: Boolean = false
)