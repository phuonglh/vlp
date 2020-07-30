package vlp.nli

case class ConfigTeller(
    master: String = "local[*]",
    executorMemory: String = "8g",
    language: String = "vi",
    mode: String = "train",
    maxSequenceLength: Int = 40,
    minFrequency: Double = 1.0,
    numFeatures: Int = 32768,
    encoder: String = "cnn",
    embeddingSize: Int = 100,
    encoderOutputSize: Int = 128,
    numLabels: Int = 3,
    modelType: String = "seq",
    modelPath: String = "dat/nli/",
    dataPath: String = "dat/nli/vi.jsonl",
    batchSize: Int = 32,
    epochs: Int = 30,
    learningRate: Double = 0.001
)