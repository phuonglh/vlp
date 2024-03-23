package vlp.vsc

/**
 * @param hiddenSize size of the encoder layers (embeddingSize)
 * @param nBlock number of blocks
 * @param nHead number of heads
 * @param maxPositionLen sequence length, default is 512
 * @param intermediateSize The size of the "intermediate" (i.e., feed-forward)
 * @param hiddenPDrop The dropout probability for all fully connected layers
 * @param attnPDrop drop probability of attention
*/
case class ConfigBERT(
  hiddenSize: Int = 48,        // 16 times less than the default 768
  nBlock: Int = 2,             // default is 12
  nHead: Int = 2,              // default is 12
  maxPositionLen: Int = 60,    // should be the same as config.maxSeqLen
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
  executorCores: Int = 8, // Y
  executorMemory: String = "8g", // Z
  driverMemory: String = "8g", // D
  mode: String = "eval",
  minFrequency: Int = 1,
  vocabSize: Int = 32768,
  percentage: Double = 1.0, // percentage of the training data set to use
  modelPath: String = "bin",
  batchSize: Int = 128,
  maxSequenceLength: Int = 60,
  learningRate: Double = 5E-4,
  epochs: Int = 80,
  embeddingSize: Int = 64,
  layers: Int = 1, // number of recurrent layers
  recurrentSize: Int = 64, // number of units in a recurrent layer
  dropoutProbability: Double = 0.0,
  language: String = "vietnamese",
  inputPath: String = "dat/vsc/vud.txt.inp",
  outputPath: String = "out/vsc/output.txt", 
  scorePath: String = "dat/vsc/scores.ged.json",
  verbose: Boolean = false,
  ged: Boolean = false, // multi-language GED data
  modelType: String = "tk",
  bert: ConfigBERT = ConfigBERT()
) extends Serializable

