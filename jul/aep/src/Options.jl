
# Vietnamese dependency treebank
options = Dict{Symbol,Any}(
    :mode => :train,
    :minFreq => 2,
    :lowercase => true,
    :maxSequenceLength => 40,
    :featuresPerContext => 4,
    :vocabSize => 2^16,
    :wordSize => 16,
    :shapeSize => 4,
    :posSize => 8, 
    :hiddenSize => 64,
    :batchSize => 32,
    :numEpochs => 20,
    :trainCorpus => string(pwd(), "/dat/dep/vie/vi-ud-train.conllu"),
    :devCorpus => string(pwd(), "/dat/dep/vie/vi-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/vie/vi-ud-dev.conllu"),
    :modelPath => string(pwd(), "/jul/aep/dat/vie/mlp.bson"),
    :wordPath => string(pwd(), "/jul/aep/dat/vie/words.txt"),
    :shapePath => string(pwd(), "/jul/aep/dat/vie/shapes.txt"),
    :posPath => string(pwd(), "/jul/aep/dat/vie/partOfSpeech.txt"),
    :labelPath => string(pwd(), "/jul/aep/dat/vie/label.txt"),
    :numCores => 4,
    :verbose => false,
    :gpu => false,
    :logPath => string(pwd(), "/jul/aep/dat/vie/loss.txt"),
    :unknown => "[unk]",
    :padding => "[pad]"
)

# English Web Treebank corpus
optionsEWT = Dict{Symbol,Any}(
    :mode => :train,
    :minFreq => 2,
    :lowercase => true,
    :maxSequenceLength => 40,
    :featuresPerContext => 4,
    :vocabSize => 2^16,
    :embeddingSize => 100,
    :hiddenSize => 64,
    :batchSize => 32,
    :numEpochs => 50,
    :trainCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-train.conllu"),
    :devCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-test.conllu"),
    :modelPath => string(pwd(), "/jul/aep/dat/eng/mlp.bson"),
    :wordPath => string(pwd(), "/jul/aep/dat/eng/words.txt"),
    :shapePath => string(pwd(), "/jul/aep/dat/eng/shapes.txt"),
    :posPath => string(pwd(), "/jul/aep/dat/eng/partOfSpeech.txt"),
    :labelPath => string(pwd(), "/jul/aep/dat/eng/label.txt"),
    :numCores => 4,
    :verbose => false,
    :gpu => false,
    :logPath => string(pwd(), "/jul/aep/dat/eng/loss.txt"),
    :unknown => "[unk]",
    :padding => "[pad]"
)

