package vlp.nli

import java.nio.file.Paths

class DataPack(val name: String, val language: String) {
  
  def dataPath(tokenized: Boolean) = language match {
    case "vi" => if (tokenized) ("dat/nli/XNLI-1.0/vi.tok.jsonl", "dat/nli/XNLI-1.0/vi.tok.jsonl", "dat/nli/XNLI-1.0/vi.tok.jsonl")
      else ("dat/nli/XNLI-1.0/vi.jsonl", "dat/nli/XNLI-1.0/vi.jsonl", "dat/nli/XNLI-1.0/vi.jsonl")
    case "en" => name match {
      case "xnli" => ("dat/nli/XNLI-1.0/en.jsonl", "dat/nli/XNLI-1.0/en.jsonl", "dat/nli/XNLI-1.0/en.jsonl")
      case "snli" => ("dat/nli/SNLI-1.0/snli_1.0_train.jsonl", "dat/nli/SNLI-1.0/snli_1.0_dev.jsonl", "dat/nli/SNLI-1.0/snli_1.0_test.jsonl")
      case _ => ("", "", "")
    }
    case _ => ("", "", "")
  }

  def modelPath() = Paths.get("bin/nli/", name, language).toString()
}
