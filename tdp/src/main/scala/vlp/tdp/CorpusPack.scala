package vlp.tdp

/**
  * Created by phuonglh on 6/27/17.
  * 
  */
class CorpusPack(val language: Language.Value = Language.Vietnamese) extends Serializable {
  
  def modelPath: String = "dat/tdp/" + lang + "/"

  def lang: String = {
    language match {
      case Language.Vietnamese => "vie"
      case Language.English => "eng"
    }
  }
  
  def dataPaths: (String, String) = {
    language match {
      case Language.Vietnamese => ("dat/dep/vie/vi-ud-train.conllu", "dat/dep/vie/vi-ud-dev.conllu")
      case Language.English => ("dat/dep/eng/tag/train.txt", "dat/dep/eng/tag/dev.txt")
    }
  }
}
