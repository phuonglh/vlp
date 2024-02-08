package vlp.dep

case class ConfigDEP(
    master: String = "local[*]",
    totalCores: Int = 8,    // X
    executorCores: Int = 4, // Y ==> there are X/Y executors
    executorMemory: String = "6g", // Z
    driverMemory: String = "16g", // D
    mode: String = "eval",
    maxVocabSize: Int = 32768,
    tokenEmbeddingSize: Int = 32, // 100
    tokenHiddenSize: Int = 32,
    partsOfSpeechEmbeddingSize: Int = 25,
    maxCharLen: Int = 13,
    charEmbeddingSize: Int = 32,
    charHiddenSize: Int = 16,
    layers: Int = 2, // number of LSTM layers or Transformer blocks
    batchSize: Int = 128,
    maxSeqLen: Int = 30,
    epochs: Int = 100,
    learningRate: Double = 5E-3,
    language: String = "eng", // [mul, vie, ind, eng]
    modelPath: String = "bin/dep",
    //    trainPaths: Seq[String] = Seq("dat/dep/vie/UD_Vietnamese-VTB/vi_vtb-ud-train.conllu", "dat/dep/ind/UD_Indonesian-GSD/id_gsd-ud-train.conllu"),
    //    validPaths: Seq[String] = Seq("dat/dep/vie/UD_Vietnamese-VTB/vi_vtb-ud-dev.conllu", "dat/dep/ind/UD_Indonesian-GSD/id_gsd-ud-dev.conllu"),
    //    trainPaths: Seq[String] = Seq("dat/dep/vie/UD_Vietnamese-VTB/vi_vtb-ud-train.conllu"),
    //    validPaths: Seq[String] = Seq("dat/dep/vie/UD_Vietnamese-VTB/vi_vtb-ud-dev.conllu"),
    trainPaths: Seq[String] = Seq("dat/dep/eng/UD_English-EWT/en_ewt-ud-train.conllu"),
    validPaths: Seq[String] = Seq("dat/dep/eng/UD_English-EWT/en_ewt-ud-dev.conllu"),
    //        trainPaths: Seq[String] = Seq("dat/dep/ind/UD_Indonesian-GSD/id_gsd-ud-train.conllu"),
    //        validPaths: Seq[String] = Seq("dat/dep/ind/UD_Indonesian-GSD/id_gsd-ud-dev.conllu"),
    testPaths: Seq[String] = Seq("dat/dep/vie/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu", "dat/dep/ind/UD_Indonesian-GSD/id_gsd-ud-test.conllu"),
    outputPath: String = "out/dep/",
    scorePath: String = "dat/dep/scores.json",
    modelType: String = "t+c", // [t, c, @c, t+c, t+p, b]
    gloveFile: String = "dat/emb/glove.6B.100d.txt",
    numberbatchFile: String = "dat/emb/numberbatch-en-19.08.txt"
)

