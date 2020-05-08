package vlp.ner


case class ConfigNER(
  master: String = "local[*]",
  mode: String = "eval",
  verbose: Boolean = false,
  modelPath: String = "dat/ner/",
  language: String = "vie",
  dataPath: String = "dat/ner/vie/vie.train",
  lambda: Double = 1E-6,
  numFeatures: Int = 32768,
  iterations: Int = 300,
  independent: Boolean = false,
  reversed: Boolean = false,
  input: String = "input.txt",
  output: String = "output.txt"
)
