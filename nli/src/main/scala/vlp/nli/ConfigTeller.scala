package vlp.nli

case class ConfigTeller(
    master: String = "local[1]",
    executorMemory: String = "8g",
    language: String = "vi",
    mode: String = "eval",
    maxSequenceLength: Int = 40,
    minFrequency: Double = 2.0,
    numFeatures: Int = 32768,
    encoder: String = "cnn",
    embeddingSize: Int = 100,
    encoderOutputSize: Int = 128,
    kernelWidth: Int = 5, // kernel with when using CNN
    numLabels: Int = 3,
    modelType: String = "seq",
    modelPath: String = "dat/nli/",
    dataPath: String = "dat/nli/vi.jsonl",
    batchSize: Int = 32,
    epochs: Int = 30,
    learningRate: Double = 0.001
)