package vlp.nli

case class ConfigTeller(
    master: String = "local[*]",
    executorMemory: String = "8g",
    language: String = "vi",
    mode: String = "eval",
    maxSequenceLength: Int = 40,
    minFrequency: Double = 2.0,
    numFeatures: Int = 32768,
    modelType: String = "bow",
    encoderType: String = "cnn",
    embeddingSize: Int = 100,
    encoderOutputSize: Int = 128,
    filterSize: Int = 5, // filter size when using CNN
    numLabels: Int = 3,
    modelPath: String = "dat/nli/",
    dataPath: String = "dat/nli/vi.jsonl",
    batchSize: Int = 64,
    epochs: Int = 50,
    learningRate: Double = 0.001,
    verbose: Boolean = false
)