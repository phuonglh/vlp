# phuonglh
# Options used for ScRNN model.

options = Dict{Symbol,Any}(
    :minFrequency => 1,
    :embeddingSize => 16,
    :hiddenSize => 64,
    :numEpochs => 20,
    :batchSize => 32,
    :labels => [:n, :s, :r, :i, :d],
    :inputPath => string(pwd(), "/dat/vsc/200.txt.inp"),
    :outputPath => string(pwd(), "/dat/vsc/200.txt.out"),
    :modelPath => string(pwd(), "/jul/vsc/dat/200.bson"),
    :gpu => false,
    :verbose => false
)

