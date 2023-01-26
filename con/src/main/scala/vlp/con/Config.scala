package vlp.con

/**
 * @param hiddenSize size of the encoder layers (embeddingSize)
 * @param nBlock block number
 * @param nHead head number
 * @param maxPositionLen sequence length, default is 512
 * @param intermediateSize The size of the "intermediate" (i.e., feed-forward)
 * @param hiddenPDrop The dropout probability for all fully connected layers
 * @param attnPDrop drop probability of attention
*/
case class ConfigBERT(
  hiddenSize: Int = 48,        // 16 times less than the default 768
  nBlock: Int = 2,             // default is 12
  nHead: Int = 2,              // default is 12
  maxPositionLen: Int = 30,    // should be the same as config.maxSeqLen
  intermediateSize: Int = 48,  // 64 times less than the default 3072
  hiddenPDrop: Double = 0.1,   // default value of BigDL BERT
  attnPDrop: Double = 0.1      // default value of BigDL BERT
) extends Serializable 

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
  epochs: Int = 5,
  embeddingSize: Int = 16,
  layers: Int = 1, // number of bi-recurrent layers
  recurrentSize: Int = 64, // number of units in a recurrent layer
  hiddenSize: Int = 32, // number of units in the last dense layer
  dropoutProbability: Double = 0.1,
  inputPath: String = "dat/vsc/100.txt.inp",
  outputPath: String = "dat/vsc/100.txt.out", 
  scorePath: String = "dat/scores.json",
  verbose: Boolean = false,
  modelType: String = "tb", // tk: token LSTM, ch: character LSTM, tb: token BERT, cb: character BERT
  bert: ConfigBERT = ConfigBERT()
) extends Serializable

