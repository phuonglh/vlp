package vlp.con

/**
  * phuonglh@gmail.com
  *
  */
case class Config(
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y ==> there is Y/X executors 
  executorMemory: String = "8g", // Z
  driverMemory: String = "8g", // D
  mode: String = "eval",
  minFrequency: Int = 1,
  vocabSize: Int = 32768,
  percentage: Double = 1.0, // percentage of the training data set to use
  modelPath: String = "bin",
  batchSize: Int = 64,
  maxSequenceLength: Int = 30,
  learningRate: Double = 5E-4,
  epochs: Int = 40,
  embeddingSize: Int = 16,
  layers: Int = 1, // number of bi-recurrent layers
  recurrentSize: Int = 128, // number of units in a recurrent layer
  hiddenSize: Int = 64, // number of units in the dense layer
  dropoutProbability: Double = 0.1,
  inputPath: String = "dat/vsc/vud.txt.inp",
  outputPath: String = "dat/vsc/vud.txt.out", 
  scorePath: String = "dat/scores.json",
  verbose: Boolean = false,
  modelType: String = "ch", // tk: token, ch: character
  configBERT: ConfigBERT = ConfigBERT()
) extends Serializable
