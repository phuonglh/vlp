"""
    RNN-based named-entity classifier.
    phuonglh
    November 8, 2019
"""

using Flux
using Flux: onehotbatch, onehot, onecold
using Flux: @epochs
using Statistics: mean
using BSON: @save, @load

include("Corpus.jl")

# Some constants
prefix = string(homedir(), "/group.vlp/")
minFreq = 2
makeLowercase = true
maxSequenceLength = 50

# Read training sentences
corpus = readCoNLL(string(prefix, "dat/ner/vie.train"))
# filter sentences of length not greater than maxSequenceLength
sentences = filter(s -> length(s.tokens) <= maxSequenceLength, corpus)
# Read test sentences
corpus_test = readCoNLL(string(prefix, "dat/ner/vie.test"))
sentences_test = filter(s -> length(s.tokens) <= maxSequenceLength, corpus_test)

include("Featurization.jl")

(xs, ys) = vectorize(sentences[1], true)
println(size(xs))
println(size(ys))

# input dimension and output dimension
inp = wd + length(partsOfSpeech) + length(chunkTypes) + length(wordShapes)
out = length(entities)
# hidden dimension of the RNN
hid = 16

# Create a model
model = Chain(
    Dense(inp, 128, relu),
    GRU(128, hid),
    Dense(hid, out),
    softmax
)

# Loss function on a data point
function loss(x, y)
    v = Flux.crossentropy(model(x), y)
    Flux.reset!(model)
    return v
end

# Loss function on a training dataset
function loss(dataset)
    values = map(p -> loss(p[1], p[2]), dataset)
    reduce(+, Tracker.data.(values))
end

accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

zs = model(xs)
println("size(zs) = ", size(zs))

# dataset is an array of [(x_1, y_1), (x_2, y_2,),... ]
println("Vectorizing the dataset... Please wait.")
@time dataset = map(s -> vectorize(s, true), sentences)
println("#(dataset) = ", length(dataset))

# train the model with some number of epochs and save the parameters to a BSON file
function train(numEpochs::Int, modelPath::String)
    Flux.reset!(model)
    optimizer = RMSProp()
    evalcb = () -> @show(loss(dataset))
    @epochs numEpochs Flux.train!(loss, params(model), dataset, optimizer, cb = Flux.throttle(evalcb, 30))
    theta = Tracker.data.(params(model))
    @save modelPath theta
end

# Predicts the label sequence of a sentence
function predict(sentence::Sentence)::Array{String}
    Flux.reset!(model)
    x = vectorize(sentence)
    y = onecold(model(x))
    map(e -> entities[e], y)
end

# Predicts a list of test sentences and outputs to the prediction 
# result to an output file in the CoNLL evaluation format
function predict(sentences::Array{Sentence}, outputPath::String)
    zs = map(s -> predict(s), sentences)
    ys = map(s -> map(token -> token.properties['e'], s.tokens), sentences)
    file = open(outputPath, "w")
    for i = 1:length(ys)
        (y, z) = (ys[i], zs[i])
        for j = 1:length(y)
            write(file, string(y[j], " ", z[j], "\n"))
        end
        write(file, "\n")
    end
    close(file)
end

@time train(15, string(prefix, "dat/ner/vie.bson"))
predict(sentences, string(prefix, "dat/ner/vie.train.flux.out"))
predict(sentences_test, string(prefix, "dat/ner/vie.test.flux.out"))