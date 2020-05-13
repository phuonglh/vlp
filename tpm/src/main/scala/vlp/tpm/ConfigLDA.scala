package vlp.tpm

/**
  * phuonglh, 5/28/18, 1:04 PM
  */
case class ConfigLDA(
  master: String = "local[*]",
  memory: String = "8g",
  mode: String = "eval",
  verbose: Boolean = false,
  minFrequency: Int = 3,
  modelPath: String = "dat/tpm",
  k: Int = 50,
  top: Int = 20,
  numFeatures: Int = 32768,
  iterations: Int = 300,
  dataPath: String = "dat/txt/fin.json"
)
