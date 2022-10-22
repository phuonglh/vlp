package vlp.tcl

/**
* phuonglh, 5/28/18, 1:04 PM
*/
case class ConfigTCL(
  master: String = "local[*]",
  executorMemory: String = "8g",
  mode: String = "eval",
  verbose: Boolean = false,
  minFrequency: Int = 3,
  modelPath: String = "dat/tcl/",
  lambda: Double = 1E-4,
  classifier: String = "mlr",
  numFeatures: Int = 65536,
  hiddenUnits: String = "128",
  numTrees: Int = 256, // number of trees used in the RFC
  maxDepth: Int = 15, // max tree depth used in the RFC
  batchSize: Int = 32,
  iterations: Int = 300,
  dataPath: String = "/home/phuonglh/Downloads/Upwork/output.csv", // "dat/vne/5catsSample/part-*",
  percentage: Double = 1.0, // percentage of the training data to be used
  inputColumnName: String = "description",
  maxTokenLength: Int = 50,
  input: String = "dat/shi/vi.txt",
  output: String = "dat/shi/vi.json"
)
