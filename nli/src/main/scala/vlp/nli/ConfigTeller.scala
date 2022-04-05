package vlp.nli

case class ConfigTeller(
    master: String = "local[*]",
    totalCores: Int = 8,
    executorCores: Int = 8,
    executorMemory: String = "8g",
    language: String = "vi",
    mode: String = "eval",
    maxSequenceLength: Int = 40,
    minFrequency: Double = 1.0,
    numFeatures: Int = 32768,
    modelType: String = "bow",
    encoderType: String = "NA", // use cnn/gru for seq or par model
    embeddingSize: Int = 25,
    encoderOutputSize: Int = 128,
    bidirectional: Boolean = false, // use with GRU
    filterSize: Int = 5, // use with CNN
    numLabels: Int = 3,
    dataPack: String = "xnli",
    batchSize: Int = 32,
    epochs: Int = 40,
    learningRate: Double = 0.001, // should use a smaller one for BERT
    verbose: Boolean = false,
    tokenized: Boolean = false,
    numBlocks: Int = 2, // use with BERT
    numHeads: Int = 4, // use with BERT
    intermediateSize: Int = 128 // use with BERT
)