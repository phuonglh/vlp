package vlp.ner


case class ConfigNER(
  master: String = "local[*]",
  executorMemory: String = "8g",
  mode: String = "eval",
  verbose: Boolean = false,
  modelPath: String = "dat/ner/",
  language: String = "vie",
  dataPath: String = "dat/ner/two/train.txt",
  validationPath: String = "dat/ner/two/dev.txt",
  lambda: Double = 1E-6,
  numFeatures: Int = 32768,
  minFrequency: Int = 2,
  iterations: Int = 300,
  independent: Boolean = false,
  reversed: Boolean = false,
  input: String = "dat/ner/vie/input.txt",
  maxSequenceLength: Int = 80,
  batchSize: Int = 128,
  wordEmbeddingSize: Int = 100,
  shapeEmbeddingSize: Int = 10,
  recurrentSize: Int = 64,
  outputSize: Int = 0,
  epochs: Int = 50,
  bidirectional: Boolean = false,
  twoColumns: Boolean = false
)
