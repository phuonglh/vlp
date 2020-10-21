package vlp.vdg

/**
  * phuonglh
  *
  * @param master Apache Spark master URL
  * @param totalCores total number of cores of all the machines
  * @param executorCores number of cores for each executors
  * @param driverMemory memory for the driver
  * @param executorMemory memory for each executor
  * @param mode running mode of the program (train/eval/exp)
  * @param minFrequency feature frequency cutoff
  * @param jsonData use this only if the input data is in JSON format
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
  * @param peephole use peep hole
  * @param hiddenUnits number of hidden units used in a recurrent layer
  * @param dropout dropout rate
  * @param inputPath path to an input file 
  * @param logPath path to a log file which contain scores (scores.json)
  * @param modelType either model 1, or 2 or 3.
  * @param lookupWordSize word embedding size
  * @param lookupCharacterSize chacter embedding size
  * @param delimiters punctutations and delimiters
  * @param verbose verbose mode or not
  */
case class ConfigVDG(
  master: String = "local[*]",
  totalCores: Int = 8,
  executorCores: Int = 8,
  driverMemory: String = "8g",
  executorMemory: String = "8g",
  mode: String = "eval",
  minFrequency: Int = 3,
  jsonData: Boolean = false,
  dataPath: String = "dat/txt/vtb.txt",
  percentage: Double = 1.0, 
  modelPath: String = "dat/vdg/",
  batchSize: Int = 32,
  epsilon: Double = 1E-5,
  maxSequenceLength: Int = 100,
  learningRate: Double = 0.001,
  epochs: Int = 15,
  layers: Int = 1, // number of bi-recurrent layers
  gru: Boolean = true, // use GRU or LSTM, default is GRU
  peephole: Boolean = false,
  hiddenUnits: Int = 64, // number of hidden units in each layer
  dropout: Double = 0,
  inputPath: String = "dat/vdg/test.txt",
  logPath: String = "dat/vdg/scores.json",
  modelType: Int = 1,
  lookupWordSize: Int = 300,
  lookupCharacterSize: Int = 30,
  delimiters: String = """[\s.,/;?!:'"…”“’+̀= ́&)(|‘– ̃ ̉•_><*̛̆©̂@ð°ö​#²®·●ñš~‎›øçî□-]+""",
  verbose: Boolean = false
) extends Serializable
