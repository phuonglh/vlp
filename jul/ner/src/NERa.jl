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
using Dates
using Random


include("Corpus.jl")
include("WordVectors.jl")
include("Embedding.jl")

# Some constants
prefix = string(homedir(), "/vlp/")
minFreq = 2
maxSequenceLength = 30
numEpochs = 10
# word vector dimension
wd = 50
makeLowercase = true
# hidden dimension of the RNN
hid = 64
# use GPU or not
g = false

# 1. Read training sentences
corpus = readCoNLL(string(prefix, "dat/ner/vie/vie.train"))
# filter sentences of length not greater than maxSequenceLength
sentences = filter(s -> length(s.tokens) <= maxSequenceLength, corpus)
println("#(sentences of length <= $(maxSequenceLength)) = ", length(sentences))

# 2. Featurize the dataset

# build vocab and embedding matrix
wordList = vocab(sentences, minFreq, makeLowercase)
push!(wordList, "<number>")
push!(wordList, "UNK")
println("#(vocab) = ", length(wordList))

# # build word vectors
# @time wordVectors = load("/opt/data/emb/vi/skip.vie.50d.txt")

# # prepare the word embedding table
N = length(wordList)
# W = randn(wd, N) * 0.01
# for i = 1:N
#     word = wordList[i]
#     if (haskey(wordVectors, word))
#         W[:, i] = wordVectors[word]
#     end
# end
# embed = Embedding(W)

embed = Embedding(N, wd)

@info "initial total weight of word embedding = $(sum(embed.W))"

# build a word index (word => index)
wordIndex = Dict{String, Int}(word => i for (i, word) in enumerate(wordList))

entities = labels(sentences, 'e')
prepend!(entities, ["BOS", "EOS"])
println("Entity types = ", entities)

partsOfSpeech = labels(sentences, 'p')
chunkTypes = labels(sentences, 'c')
wordShapes = labels(sentences, 's')

# NOTE: should make word embedding learnable by not encoding vectors into Xs!

"""
Creates a matrix representation of a sentence. Each sequence of N tokens 
is converted to a matrix of size DxN, each column vector coressponds to a token.
"""
function vectorize(sentence::Sentence, training::Bool = false)
    tokens = sentence.tokens
    # word indices for embeddings
    ws = Array{Int,1}()
    for i = 1:length(tokens)
        word = lowercase(tokens[i].text)
        if (haskey(wordIndex, word))
            push!(ws, wordIndex[word])
        elseif (shape(word) == "number")
            push!(ws, wordIndex["<number>"])
        else
            push!(ws, wordIndex["UNK"])
        end
    end
    # one-hot parts-of-speech vector
    pos = map(token -> token.properties['p'], tokens)
    ps = onehotbatch(pos, partsOfSpeech)
    # one-hot chunk type vector
    chs = map(token -> token.properties['c'], tokens)
    cs = onehotbatch(chs, chunkTypes)
    # one-hot shape vector
    shs = map(token -> token.properties['s'], tokens)
    ss = onehotbatch(shs, wordShapes)
    # convert xs to Float32 to speed up computation
    xs = Float32.(vcat(embed(ws), ps, cs, ss))
    if (training)
        yy = map(token -> token.properties['e'], tokens)
        # padding a start symbol
        ys0 = onehotbatch(["BOS", yy...], entities)
        # padding an end symbol
        yss = onehotbatch([yy..., "EOS"], entities)
        (xs, Float32.(ys0), Float32.(yss))
    else
        xs
    end    
end

(xs, ys) = vectorize(sentences[1], true)
println(size(xs))
println(size(ys))

# 3. Build the model
# input dimension and output dimension
inp = wd + length(partsOfSpeech) + length(chunkTypes) + length(wordShapes)
out = length(entities)

# Encoder: a projection layer followed by a bi-directional RNN to encode input sequences 
# 
forward  = GRU(inp, hid ÷ 2)
backward = GRU(inp, hid ÷ 2)
# tokens is a batch of sequences, each sequence is a 2-d matrix representing 
# an input sentence, each column corresponds to a word. That is, `tokens` is an  
# array of (hid x N) matrices, where N is the same length of sentences in the batch.
encode(tokens) = vcat.(forward.(tokens), flip(backward, tokens))

# create an alignment model
alignNet = Dense(2*hid, 1)
# s is an output of the decoder, h is a hidden state of input at a position j.
function align(s, h)
    a = if length(size(s)) == 1 # if recur is reset
        repeat(s, 1, size(h, 2))
    else
        s
    end
    alignNet(vcat(h, a))
end

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
state = (embed, forward, backward, alignNet, recur, toAlpha)

function model(x, y)
    prediction = decode(encode(x), y)
    reset!(state)
    return prediction
end

# Loss function on a batch. For each sequence in the batch, 
# we apply the model and compute the cross-entropy element-wise.
# The total loss of the batch is returned.
loss(xb, yb0, yb) = sum(crossentropy.(model(xb, yb0), yb))

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
batches(xs, p) = [batchseq(b, p) for b in partition(xs, maxSequenceLength)]
# batch Xs with a padding vector of zeros
Xb = batches(Xs, Float32.(zeros(inp)))
# batch Ys with a pading vector of one-hot EOS
Yb0 = batches(Ys0, onehot("EOS", entities))
# batch Ys with a padding vector one one-hot EOS
Yb = batches(Ys, onehot("EOS", entities))
# create a data set for training, each training point is a tuple of batches
dataset = collect(zip(Xb, Yb0, Yb))
println("#(batches) = ", length(dataset))

println("typeof(Xb10) = ", typeof(Xb[10])) # this should be Array{Array{Float32,2},1}

# train the model with some number of epochs and save the parameters to a BSON file
function train(numEpochs::Int, modelPath::String)
    evalcb = throttle(600) do
        J = loss(dataset[10]...)
        @show(J)
        @save "$(modelPath)/vie-$(now()).bson" state loss = J
    end
    optimizer = ADAM(1E-4)
    @epochs numEpochs Flux.train!(loss, params(state), dataset, optimizer, cb = evalcb)
end

# Predicts the label sequence of a sentence
# TODO: Should implement the beam search method to find the best sequence
function predict(sentence::Sentence)::Array{String}
    reset!(state)
    xs = vectorize(sentence)
    # flatten xs and encode the resulting sequence, one by one.
    ts = encode(flatten(xs))
    ps = ["BOS"]
    for i = 1:length(ts)
        dist = decode1(ts, onehot(ps[end], entities))
        next = wsample(entities, vec(dist))
        push!(ps, next)
    end
    return ps[2:end]
end

# Predicts a list of test sentences and outputs to the prediction 
# result to an output file in the CoNLL evaluation format
function predict(sentences::Array{Sentence}, outputPath::String)
    println("Predicting sentences. Please wait...")
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
    println("Done. Prediction result is written to ", outputPath)
end

@time train(numEpochs, string(prefix, "dat/ner/vie"))

@info "final total weight of word embedding = $(sum(embed.W))"

@time predict(sentences, string(prefix, "dat/ner/vie/vie.train.jul.nerA.", hid))

# Read test sentences
# corpus_test = readCoNLL(string(prefix, "dat/ner/vie/vie.test"))
# sentences_test = filter(s -> length(s.tokens) <= maxSequenceLength, corpus_test)
# println("#(sentences_test) = ", length(sentences_test))
#@time predict(sentences_test, string(prefix, "dat/ner/vie/vie.test.jul.nerA.", hid))

