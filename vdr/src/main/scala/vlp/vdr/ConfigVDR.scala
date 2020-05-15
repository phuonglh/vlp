package vlp.vdr

case class ConfigVDR(
  master: String = "local[*]",
  mode: String = "eval",
  verbose: Boolean = false,
  modelPath: String = "dat/vdr/",
  dataPath: String = "dat/txt/vtb.txt",
  mappingResourcePath: String = "/mappings.txt",
  markovOrder: Int = 2,
  lambda: Double = 1E-6,
  numFeatures: Int = 65536,
  iterations: Int = 400,
  greedy: Boolean = false,
  input: String = ""
)
