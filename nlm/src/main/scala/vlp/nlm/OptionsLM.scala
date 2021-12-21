package vlp.nlm
/**
  * phuonglh@gmail.com
  *
  */
final case class OptionsLM(
    master: String = "local[*]",
    executorMemory: String = "8g",
    mode: String = "train",
    trainDataPath: String = "dat/txt/vlsp.jul.tok",
    validDataPath: String = "dat/txt/vlsp.jul.tok",
    dictionaryPath: String = "dat/nlm",
    vocabSize: Int = 20000,
    numSteps: Int = 20,
    batchSize: Int = 128,
    numHeads: Int = 4,
    maxEpoch: Int = 100,
    modelType: String = "rnn", // ["rnn", "trm"]
    numLayers: Int = 2,
    hiddenSize: Int = 256,
    keepProb: Float = 0.2f,
    checkpoint: Option[String] = Some("dat/nlm"),
    overWriteCheckpoint: Boolean = true,
    learningRate: Double = 1E-3,
    verbose: Boolean = false
)

