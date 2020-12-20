
# Vietnamese dependency treebank
options = Dict{Symbol,Any}(
    :mode => :train,
    :minFreq => 2,
    :lowercase => true,
    :featuresPerContext => 20,
    :numFeatures => 2^16,
    :embeddingSize => 50,
    :hiddenSize => 128,
    :batchSize => 32,
    :numEpochs => 50,
    :trainCorpus => string(pwd(), "/dat/dep/vie/vi-ud-train.conllu"),
    :devCorpus => string(pwd(), "/dat/dep/vie/vi-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/vie/vi-ud-dev.conllu"),
    :modelPath => string(pwd(), "/jul/tdp/dat/vie/mlp.bson"),
    :vocabPath => string(pwd(), "/jul/tdp/dat/vie/vocab.txt"),
    :labelPath => string(pwd(), "/jul/tdp/dat/vie/label.txt"),
    :numCores => 4,
    :verbose => false
)

# English Web Treebank corpus
optionsEWT = Dict{Symbol,Any}(
    :mode => :train,
    :minFreq => 2,
    :lowercase => true,
    :featuresPerContext => 20,
    :numFeatures => 2^16,
    :embeddingSize => 50,
    :hiddenSize => 128,
    :batchSize => 32,
    :numEpochs => 50,
    :trainCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-dev.conllu"),
    :devCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-test.conllu"),
    :modelPath => string(pwd(), "/jul/tdp/dat/eng/mlp.bson"),
    :vocabPath => string(pwd(), "/jul/tdp/dat/eng/vocab.txt"),
    :labelPath => string(pwd(), "/jul/tdp/dat/eng/label.txt"),
    :numCores => 4,
    :verbose => false
)

