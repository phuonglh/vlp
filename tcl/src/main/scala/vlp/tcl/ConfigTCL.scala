package vlp.tcl

/**
* phuonglh, 5/28/18, 1:04 PM
*/
case class ConfigTCL(
  master: String = "local[*]",
  mode: String = "eval",
  verbose: Boolean = false,
  minFrequency: Int = 3,
  modelPath: String = "dat/tcl/",
  lambda: Double = 1E-4,
  classifier: String = "mlr",
  numFeatures: Int = 32768,
  hiddenUnits: Int = 16,
  layers: String = "16",
  batchSize: Int = 32,
  iterations: Int = 300,
  dataPath: String = "dat/vne/5catsSample/part-*",
  input: String = "input.txt",
  output: String = "output.txt"
)
