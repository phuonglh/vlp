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
  minFrequency: Int = 2,
  vocabSize: Int = 32768,
  percentage: Double = 1.0, // percentage of the training data set to use
  modelPath: String = "bin",
  batchSize: Int = 64,
  epsilon: Double = 1E-5,
  maxSequenceLength: Int = 30,
  learningRate: Double = 0.001,
  epochs: Int = 2,
  embeddingSize: Int = 16,
  layers: Int = 1, // number of bi-recurrent layers
  gru: Boolean = true, // use GRU or LSTM, default is GRU
  recurrentSize: Int = 16, // number of units in a recurrent layer
  hiddenSize: Int = 16, // number of units in the dense layer
  dropoutProbability: Double = 0.1,
  inputPath: String = "dat/vsc/vud.txt.inp",
  outputPath: String = "dat/vsc/vud.txt.out", 
  scorePath: String = "dat/scores.json",
  delimiters: String = """[\s.,/;?!:'"…”“’+̀= ́&)(|‘– ̃ ̉•_><*̛̆©̂@ð°ö​#²®·●ñš~‎›øçî□-]+""",
  verbose: Boolean = false,
  modelType: String = "sc" // tk: token, sc: semi-character
) extends Serializable
