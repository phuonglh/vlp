package vlp.ner


case class ConfigNER(
  master: String = "local[*]",
  executorMemory: String = "8g",
  mode: String = "eval",
  verbose: Boolean = false,
  modelPath: String = "dat/ner/",
  language: String = "vie",
  dataPath: String = "dat/ner/vie/vie.train",
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
  wordEmbeddingSize: Int = 200,
  shapeEmbeddingSize: Int = 10,
  recurrentSize: Int = 64,
  outputSize: Int = 0,
  epochs: Int = 80,
  bidirectional: Boolean = false
)
