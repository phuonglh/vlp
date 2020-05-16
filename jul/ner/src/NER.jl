"""
    RNN-based named-entity classifier.
    phuonglh
    November 8, 2019
"""

using Flux
using Flux: onehotbatch, onehot, onecold, batchseq, reset!, crossentropy, throttle
using Base.Iterators: partition
using Flux: @epochs
using Statistics: mean
using BSON: @save, @load
using Distributed

include("Corpus.jl")

# Some constants
prefix = string(homedir(), "/group.vlp/")
minFreq = 2
maxSequenceLength = 40

# Read training sentences
corpus = readCoNLL(string(prefix, "dat/ner/vie.train"))
# filter sentences of length not greater than maxSequenceLength
sentences = filter(s -> length(s.tokens) <= maxSequenceLength, corpus)
println("#(sentences of length <= $(maxSequenceLength)) = ", length(sentences))
# Read test sentences
corpus_test = readCoNLL(string(prefix, "dat/ner/vie.test"))
sentences_test = filter(s -> length(s.tokens) <= maxSequenceLength, corpus_test)
println("#(sentences_test) = ", length(sentences_test))

include("Featurization.jl")

(xs, ys) = vectorize(sentences[1], true)
println(size(xs))
println(size(ys))

# input dimension and output dimension
inp = wd + length(partsOfSpeech) + length(chunkTypes) + length(wordShapes)
out = length(entities)
# hidden dimension of the RNN
hid = 300

#include("BiRNN.jl")

# Create a model
model = Chain(
    #BiGRU(inp, hid),
    GRU(inp, hid),
    Dense(hid, out),
    softmax
)

# the full model which takes as input a 2-d matrix representing an input 
# sequence; each column corresponds to a token of the sequence.
function f(x)
    prediction = model(x)
    reset!(model)
    return prediction
end

# Loss function on a batch. For each sequence in the batch, 
# we apply the model and compute the cross-entropy element-wise.
# The total loss of the batch is returned.
loss(xb, yb) = sum(crossentropy.(f.(xb), yb))

batchSize = 36
println("Vectorizing the dataset... Please wait.")
# XYs is an array of samples  [(x_1, y_1), (x_2, y_2,),... ]
@time XYs = map(s -> vectorize(s, true), sentences)
# convert a 2-d array to an array of column vectors
flatten(xs) = [xs[:, i] for i = 1:size(xs,2)]
# extracts Xs and Ys
Xs = map(pair -> flatten(pair[1]), XYs)
Ys = map(pair -> flatten(pair[2]), XYs)

# batch a sequence with padding p
batches(xs, p) = [batchseq(b, p) for b in partition(xs, batchSize)]
# batch Xs with a zero vector
Xb = batches(Xs, Float32.(zeros(inp)))
# batch Ys with a zero vector
Yb = batches(Ys, Float32.(zeros(out)))
# create a data set for training, each training point is a pair of batch
dataset = collect(zip(Xb, Yb))
println("#(batches) = ", length(dataset))

X100 = Xb[100]
Y100 = Yb[100]
println("typeof(X100) = ", typeof(X100)) # this should be Array{Array{Float32,2},1}
println("typeof(Y100) = ", typeof(Y100)) # this should be Array{Array{Float32,2},1}

# train the model with some number of epochs and save the parameters to a BSON file
function train(numEpochs::Int, modelPath::String)
    optimizer = ADAM(.001)
    evalcb = () -> @show(loss(X100, Y100)) # or use loss(dataset[100]...)
    @epochs numEpochs Flux.train!(loss, params(model), dataset, optimizer, cb = throttle(evalcb, 20))
    @save modelPath model
end

# Predicts the label sequence of a sentence
function predict(sentence::Sentence)::Array{String}
    reset!(model)
    x = vectorize(sentence)
    y = onecold(model(x))
    map(e -> entities[e], y)
end

# Predicts a list of test sentences and outputs to the prediction 
# result to an output file in the CoNLL evaluation format
function predict(sentences::Array{Sentence}, outputPath::String)
    zs = pmap(s -> predict(s), sentences)
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

@time train(20, string(prefix, "dat/ner/vie.bson"))
@time predict(sentences, string(prefix, "dat/ner/vie.train.flux.out"))
@time predict(sentences_test, string(prefix, "dat/ner/vie.test.flux.out"))
