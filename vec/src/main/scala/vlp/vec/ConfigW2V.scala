package vlp.vec

/**
  * phuonglh, 5/28/18, 1:04 PM
  */

case class ConfigW2V(
  master: String = "local[*]",
  executorMemory: String = "8g",
  mode: String = "eval",
  verbose: Boolean = true,
  minFrequency: Int = 5,
  minLength: Int = 20, // min sentence length in characters
  windowSize: Int = 5,
  dimension: Int = 50,
  modelPath: String = "dat/vec/",
  iterations: Int = 50,
  input: String = "/Users/phuonglh/vlp/dat/txt/vlsp/",
  text: Boolean = false, // input data in .json format or .txt format
  output: String = "/Users/phuonglh/vlp/dat/vec/vie.50d.txt"
)
