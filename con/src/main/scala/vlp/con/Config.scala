package vlp.con

/**
  * phuonglh@gmail.com
  *
  */
case class Config(
  master: String = "local[4]",
  totalCores: Int = 4,    // X
  executorCores: Int = 4, // Y ==> there is Y/X executors 
  executorMemory: String = "8g", // Z
  driverMemory: String = "8g", // D
  mode: String = "eval",
  minFrequency: Int = 2,
  vocabSize: Int = 32768,
  percentage: Double = 0.5, // 50% of the data set to use
  modelPath: String = "bin/",
  batchSize: Int = 16,
  epsilon: Double = 1E-5,
  maxSequenceLength: Int = 20,
  learningRate: Double = 0.001,
  epochs: Int = 5,
  embeddingSize: Int = 50,
  layers: Int = 1, // number of bi-recurrent layers
  gru: Boolean = true, // use GRU or LSTM, default is GRU
  recurrentSize: Int = 64, // number of units in a recurrent layer
  hiddenSize: Int = 32, // number of units in the dense layer
  dropoutProbability: Double = 0.2,
  inputPath: String = "dat/vsc/100.txt.inp",
  outputPath: String = "dat/vsc/100.txt.out", 
  scorePath: String = "dat/scores.json",
  delimiters: String = """[\s.,/;?!:'"…”“’+̀= ́&)(|‘– ̃ ̉•_><*̛̆©̂@ð°ö​#²®·●ñš~‎›øçî□-]+""",
  verbose: Boolean = false
) extends Serializable
