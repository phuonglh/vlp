options = Dict{Symbol,Any}(
    :mode => :train,
    :minFreq => 2,
    :lowercase => true,
    :numFeatures => 2^16,
    :embeddingSize => 50,
    :hiddenSize => 64,
    :maxSequenceLength => 100,
    :batchSize => 32,
    :numEpochs => 50,
    :trainCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-train.conllu"),
    :modelPath => string(pwd(), "/jul/tdp/dat/eng/encoder.bson"),
    :vocabPath => string(pwd(), "/jul/tdp/dat/eng/vocab.txt"),
    :embeddingPath => string(pwd(), "/jul/tdp/dat/eng/embeds.txt"),
    :numCores => 4,
    :verbose => false
)