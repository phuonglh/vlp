package vlp.nli

case class ConfigTeller(
    master: String = "local[*]",
    mode: String = "eval",
    maxSequenceLength: Int = 128,
    minFrequency: Double = 2.0,
    numFeatures: Int = 8192,
    embeddingSize: Int = 100,
    outputSize: Int = 128,
    numLabels: Int = 3
)