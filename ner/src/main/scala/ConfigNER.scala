package vlp.ner


case class ConfigNER(
  master: String = "local[*]",
  executorMemory: String = "8g",
  mode: String = "eval",
  verbose: Boolean = false,
  modelPath: String = "dat/ner/",
  language: String = "vie",
  dataPath: String = "dat/ner/vie/vie.test",
  validationPath: String = "dat/ner/vie/vie.test",
  lambda: Double = 1E-6,
  numFeatures: Int = 32768,
  minFrequency: Int = 2,
  iterations: Int = 300,
  independent: Boolean = false,
  reversed: Boolean = false,
  input: String = "dat/ner/vie/input.txt",
  output: String = "dat/ner/vie/output.txt",
  maxSequenceLength: Int = 40,
  batchSize: Int = 128,
  outputSize: Int = 64,
  wordEmbeddingSize: Int = 100,
  shapeEmbeddingSize: Int = 10,
  epochs: Int = 100,
  bidirectional: Boolean = false
)
