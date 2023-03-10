import org.scalatest.flatspec.AnyFlatSpec

class PipelineTest extends AnyFlatSpec {
  "Spark NLP Starter" should "correctly download and annotate" in {
    Main.main(Array.empty)
  }

  "Spark NLP Starter" should "correctly work with pretrained Pipeline" in {
    Main.pretrainedPipeline(Array.empty)
    Main.pretrainedPipelineLD(Array.empty)
  }
}
