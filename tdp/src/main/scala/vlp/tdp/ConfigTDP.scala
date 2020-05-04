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
  minFrequency: Int = 3,
  numFeatures: Int = 1024,
  iterations: Int = 400,
  extended: Boolean = false,
  independent: Boolean = false,
  layers: String = "",
  discrete: Boolean = false // use discrete super-tag features
)