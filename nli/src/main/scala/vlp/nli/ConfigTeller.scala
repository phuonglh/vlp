package vlp.nli

case class ConfigTeller(
    master: String = "local[*]",
    mode: String = "eval",
    maxSequenceLength: Int = 40,
    minFrequency: Double = 1.0,
    numFeatures: Int = 2048,
    embeddingSize: Int = 20,
    outputSize: Int = 20,
    numLabels: Int = 3,
    modelType: String = "seq",
    modelPath: String = "dat/nli/vi",
    dataPath: String = "dat/nli/xnli.dev.vi.jsonl",
    batchSize: Int = 32,
    learningRate: Double = 0.001
)