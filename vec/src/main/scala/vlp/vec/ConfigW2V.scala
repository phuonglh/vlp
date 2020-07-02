package vlp.vec

/**
  * phuonglh, 5/28/18, 1:04 PM
  */

case class ConfigW2V(
  master: String = "local[*]",
  mode: String = "eval",
  verbose: Boolean = true,
  minFrequency: Int = 5,
  minLength: Int = 20, // min sentence length in characters
  windowSize: Int = 5,
  dimension: Int = 50,
  modelPath: String = "dat/vlp/vec/",
  iterations: Int = 50,
  data: String = "/opt/data/w2v/news.txt",
  output: String = "/opt/data/w2v/vie.50d.txt"
)
