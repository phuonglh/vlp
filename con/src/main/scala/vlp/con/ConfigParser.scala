package vlp.con

/**
  * phuonglh
  *
  * @param master Apache Spark master URL
  * @param totalCores total number of cores of all the machines
  * @param executorCores number of cores for each executors
  * @param executorMemory memory for each executor
  * @param driverMemory memory for the driver
  * @param mode running mode of the program (train/eval/exp)
  * @param minFrequency feature frequency cutoff
  * @param dataPath path to the data file (.txt or .json)
  * @param percentage use only some percentage of samples in a (big) data path 
  * @param modelPath path to the trained model
  * @param batchSize batch size
  * @param epsilon convergence tolerance
  * @param maxSequenceLength maximum sequence length
  * @param learningRate learning rate
  * @param epochs number of training epochs
  * @param layers number of layers
  * @param gru use GRU unit (if true) or LSTM unit (if false)
  * @param hiddenUnits number of hidden units used in a recurrent layer
  * @param dropout dropout rate
  * @param inputPath path to an input file 
  * @param outputPah path to the output file
  * @param scorePath path to a log file which contain scores (scores.json)
  * @param lookupWordSize word embedding size
  * @param lookupCharacterSize character embedding size
  * @param delimiters punctutations and delimiters
  * @param verbose verbose mode or not
  */
case class ConfigParser(
  master: String = "local[4]",
  totalCores: Int = 4,    // X
  executorCores: Int = 4, // Y ==> there is Y/X executors 
  executorMemory: String = "12g", // Z
  driverMemory: String = "3g", // D
  mode: String = "eval",
  minFrequency: Int = 3,
  dataPath: String = "dat/hcm.txt",
  percentage: Double = 0.5, // 50% of the data set to use
  modelPath: String = "bin/",
  batchSize: Int = 16,
  epsilon: Double = 1E-5,
  maxSequenceLength: Int = 80,
  learningRate: Double = 0.001,
  epochs: Int = 20,
  layers: Int = 1, // number of bi-recurrent layers
  gru: Boolean = true, // use GRU or LSTM, default is GRU
  hiddenUnits: Int = 64, // number of hidden units in each recurrent layer
  dropout: Double = 0,
  inputPath: String = "dat/ftel-3.txt",
  outputPath: String = "dat/ftel-3", 
  scorePath: String = "dat/scores.json",
  delimiters: String = """[\s.,/;?!:'"…”“’+̀= ́&)(|‘– ̃ ̉•_><*̛̆©̂@ð°ö​#²®·●ñš~‎›øçî□-]+""",
  verbose: Boolean = false
) extends Serializable
