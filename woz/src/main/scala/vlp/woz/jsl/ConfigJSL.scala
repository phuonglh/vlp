package vlp.woz.jsl

case class ConfigJSL(
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y ==> there are Y/X executors 
  executorMemory: String = "8g", // Z
  driverMemory: String = "8g", // D
  mode: String = "eval",
  batchSize: Int = 128,
  learningRate: Double = 5E-4,
  epochs: Int = 80,
  modelPath: String = "bin/act/",
  trainPath: String = "dat/woz/act/train/",
  devPath: String = "dat/woz/act/dev/",
  testPath: String = "dat/woz/act/test/",
  outputPath: String = "dat/out/",
  scorePath: String = "dat/woz/act/scores.json",
  modelType: String = "b", // [b, d, l, s, u, x]
) extends Serializable
