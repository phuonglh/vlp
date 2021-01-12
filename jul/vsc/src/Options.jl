# phuonglh
# Options used for ScRNN model.

options = Dict{Symbol,Any}(
    :minFreq => 1,
    :hiddenSize => 64,
    :numEpochs => 20,
    :labels => [:n, :s, :r, :i, :d],
    :dataPath = string(pwd(), "dat/vsc/013.txt.inp"),
    :modelPath => string(pwd(), "/jul/vsc/dat/scr.bson"),
    :gpu => false,
    :verbose => false
)