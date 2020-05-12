package vlp.tdp

/**
  * Created by phuonglh on 6/30/17.
  */
case class ConfigTDP
(
  master: String = "local[*]",
  memory: String = "8g",
  mode: String = "eval",
  verbose: Boolean = false,
  language: String = "vie",
  classifier: String = "mlr",
  minFrequency: Int = 2,
  numFeatures: Int = 32768,
  iterations: Int = 400,
  extended: Boolean = false, // use super-tag features
  independent: Boolean = false,
  layers: String = "",
  discrete: Boolean = true // use discrete super-tag features
)