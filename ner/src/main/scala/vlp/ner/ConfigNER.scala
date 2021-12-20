package vlp.ner


case class ConfigNER(
  master: String = "local[*]",
  executorMemory: String = "8g",
  driverMemory: String = "4g",
  mode: String = "predict",
  verbose: Boolean = false,
  modelPath: String = "dat/ner/vie",
  language: String = "vie",
  dataPath: String = "dat/ner/vie/vie.train",
  trainingPath: String = "dat/ner/vie/vie.train",
  validationPath: String = "dat/ner/vie/vie.test",
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
  epochs: Int = 100,
  bidirectional: Boolean = false,
  twoColumns: Boolean = false
)
