package vlp.woz

case class ConfigJSL(
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y ==> there are Y/X executors 
  executorMemory: String = "8g", // Z
  driverMemory: String = "16g", // D
  language: String = "en", // {en, vi}
  mode: String = "eval",
  batchSize: Int = 128,
  learningRate: Double = 5E-4,
  threshold: Float = 0.4f,
  epochs: Int = 200,
  modelPath: String = "bin/act/",
  trainPath: String = "dat/woz/act/train/",
  devPath: String = "dat/woz/act/dev/",
  validPath: String = "dat/woz/act/val", // Parquet file of devPath
  testPath: String = "dat/woz/act/test/",
  outputPath: String = "dat/out/",
  scorePath: String = "dat/scores-jsl.json",
  modelType: String = "b", // {b, u, r, x}
) extends Serializable
