"""
    Attention-based Seq2Seq model for named-entity classifier.
    This code implements an attention-based NER.
    phuonglh@gmail.com
    December 10, 2019
"""

using Flux
using Flux: onehotbatch, onehot, onecold, batchseq, reset!, crossentropy, throttle, flip
using Base.Iterators: partition
using Flux: @epochs
using Statistics: mean
using BSON: @save, @load
using StatsBase: wsample
using Distributed
using CuArrays


include("Corpus.jl")

# Some constants
prefix = string(homedir(), "/group.vlp/")
minFreq = 2
maxSequenceLength = 40

# Read training sentences
corpus = readCoNLL(string(prefix, "dat/ner/vie.test"))
# filter sentences of length not greater than maxSequenceLength
sentences = filter(s -> length(s.tokens) <= maxSequenceLength, corpus)
println("#(sentences of length <= $(maxSequenceLength)) = ", length(sentences))
# Read test sentences
corpus_test = readCoNLL(string(prefix, "dat/ner/vie.test"))
sentences_test = filter(s -> length(s.tokens) <= maxSequenceLength, corpus_test)
println("#(sentences_test) = ", length(sentences_test))

include("FeaturizationA.jl")

(xs, ys) = vectorize(sentences[1], true)
println(size(xs))
println(size(ys))

# input dimension and output dimension
inp = wd + length(partsOfSpeech) + length(chunkTypes) + length(wordShapes)
out = length(entities)
# hidden dimension of the RNN
hid = 64

# Encoder: a projection layer followed by a bi-directional RNN to encode input sequences 
# 
forward  = GRU(inp, hid÷2)
backward = GRU(inp, hid÷2)
# tokens is a batch of sequences, each sequence is a 2-d matrix representing 
# an input sentence, each column corresponds to a word. That is, `tokens` is an  
# array of (hid x N) matrices, where N is the same length of sentences in the batch.
encode(tokens) = vcat.(forward.(tokens |> gpu), flip(backward, tokens |> gpu))

# create an alignment model
alignNet = Dense(2*hid, 1) |> gpu
# s is an output of the decoder, h is a hidden state of input at a position j.
align(s, h) = alignNet(vcat(h, s .* trues(1, size(h, 2))))

# Decoder: a recurrent model which takes a sequence of annotations, attends, and returns
# a predicted output token.
recur   = LSTM(hid+out, hid)
toAlpha = Dense(hid, out)

function asoftmax(xs)
  xs = [exp.(x) for x in xs]
  s = sum(xs)
  return [x ./ s for x in xs]
end

# Decode one step: tokens is a batch of matrices
function decode1(tokens, label)
  weights = asoftmax([align(recur.state[2], t) for t in tokens])
  context = sum(map((a, b) -> a .* b, weights, tokens))
  y = recur(vcat(Float32.(label), context))
  return softmax(toAlpha(y))
end
# Decodes multiple steps
decode(tokens, labels) = [decode1(tokens, label) for label in labels]

# The full model
state = (forward, backward, alignNet, recur, toAlpha)

state = gpu.(state)

function model(x, y)
  prediction = decode(encode(x), y)
  reset!(state)
  return prediction
end

# Loss function on a batch. For each sequence in the batch, 
# we apply the model and compute the cross-entropy element-wise.
# The total loss of the batch is returned.
loss(xb, yb0, yb) = sum(crossentropy.(model(xb, yb0), yb))

batchSize = 24
println("Vectorizing the dataset... Please wait.")
# XYs is an array of samples  [(x_1, y_1), (x_2, y_2,),... ]
@time XYs = map(s -> vectorize(s, true), sentences)
# convert a 2-d array to an array of column vectors
flatten(xs) = [xs[:, i] for i = 1:size(xs,2)]
# extracts Xs and Ys
Xs = map(pair -> flatten(pair[1]), XYs)
Ys0 = map(pair -> flatten(pair[2]), XYs)
Ys = map(pair -> flatten(pair[3]), XYs)

# batch a sequence with padding p
batches(xs, p) = [batchseq(b, p) for b in partition(xs, batchSize)]
# batch Xs with a zero vector
Xb = map(t -> gpu.(t), batches(Xs, Float32.(zeros(inp))))
# batch Ys with a zero vector
Yb0 = map(t -> gpu.(t), batches(Ys0, onehot("EOS", entities)))
# batch Ys with a zero vector
Yb = map(t -> gpu.(t), batches(Ys, onehot("EOS", entities)))
# create a data set for training, each training point is a tuple of batches
dataset = collect(zip(Xb, Yb0, Yb))
println("#(batches) = ", length(dataset))

println("typeof(Xb50) = ", typeof(Xb[50])) # this should be Array{Array{Float32,2},1}

# train the model with some number of epochs and save the parameters to a BSON file
function train(numEpochs::Int, modelPath::String)
    optimizer = ADAM()
    evalcb = () -> @show(loss(dataset[50]...))
    @epochs numEpochs Flux.train!(loss, params(state), dataset, optimizer, cb = throttle(evalcb, 20))
    @save modelPath state
end

# Predicts the label sequence of a sentence
function predict(sentence::Sentence)::Array{String}
    reset!(state)
    xs = vectorize(sentence)
    # flatten xs and encode the resulting sequence, one by one.
    ts = encode(flatten(xs))
    ps = ["BOS"]
    for i = 1:length(ts)
        dist = decode1(ts, onehot(ps[end], entities))
        next = wsample(entities, vec(Tracker.data(dist)))
        push!(ps, next)
    end
    return ps[2:end]
end

# Predicts a list of test sentences and outputs to the prediction 
# result to an output file in the CoNLL evaluation format
function predict(sentences::Array{Sentence}, outputPath::String)
    println("Predicting sentences. Please wait...")
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
    println("Done. Prediction result is written to ", outputPath)
end

@time train(1, string(prefix, "dat/ner/vie.bson"))
@time predict(sentences_test, string(prefix, "dat/ner/vie.test.flux.a"))
@time predict(sentences, string(prefix, "dat/ner/vie.train.flux.a"))
