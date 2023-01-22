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
  hiddenSize: Int = 64,
  nBlock: Int = 4,
  nHead: Int = 4,
  maxPositionLen: Int = 30,
  intermediateSize: Int = 32,
  hiddenPDrop: Double = 0.1,
)
