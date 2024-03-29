package vlp.tdp

/**
  * Created by phuonglh on 6/30/17.
  */
case class ConfigTDP
(
  master: String = "local[*]",
  totalCores: Int = 8,
  executorCores: Int = 8,
  driverMemory: String = "8g",
  executorMemory: String = "8g",
  mode: String = "eval",
  verbose: Boolean = false,
  modelPath: String = "dat/tdp/",
  language: String = "vie",
  classifier: String = "mlr",
  minFrequency: Int = 2,
  numFeatures: Int = 65536,
  iterations: Int = 400,
  batchSize: Int = 32,
  extended: Boolean = false, // use super-tag features
  tagEmbeddingSize: Int = 10, // super-tag embedding size
  independent: Boolean = false,
  hiddenUnits: String = "",
  discrete: Boolean = true // use discrete super-tag features  
)