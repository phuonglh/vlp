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
include("WordVectors.jl")
include("Embedding.jl")
#include("BiRNN.jl")


# Some constants
prefix = string(homedir(), "/vlp/")
minFreq = 2
maxSequenceLength = 40
numEpochs = 20
# word vector dimension
wd = 50
makeLowercase = true
batchSize = 32
# hidden dimension of the RNN
hid = 32
# use GPU or not
g = false

# 1. Read training sentences
corpus = readCoNLL(string(prefix, "dat/ner/vie/vie.train"))
# filter sentences of length not greater than maxSequenceLength
sentences = filter(s -> length(s.tokens) <= maxSequenceLength, corpus)
println("#(sentences of length <= $(maxSequenceLength)) = ", length(sentences))
# Read test sentences
corpus_test = readCoNLL(string(prefix, "dat/ner/vie/vie.test"))
sentences_test = filter(s -> length(s.tokens) <= maxSequenceLength, corpus_test)
println("#(sentences_test) = ", length(sentences_test))

# 2. Featurize the dataset

# build vocab and embedding matrix
wordList = vocab(sentences, minFreq, makeLowercase)
push!(wordList, "<number>")
push!(wordList, "UNK")
println("#(vocab) = ", length(wordList))

# build word vectors
@time wordVectors = load("/opt/data/emb/skip.vie.50d.txt")

# prepare the word embedding table
N = length(wordList)
W = rand(wd, N)
for i = 1:N
    word = wordList[i]
    if (haskey(wordVectors, word))
        W[:, i] = wordVectors[word]
    end
end

embed = Embedding(W)

# build a word index (word => index)
wordIndex = Dict{String, Int}(word => i for (i, word) in enumerate(wordList))

entities = labels(sentences, 'e')
println("Entity types = ", entities)

partsOfSpeech = labels(sentences, 'p')
chunkTypes = labels(sentences, 'c')
wordShapes = labels(sentences, 's')

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
        ys = onehotbatch(yy, entities)
        (xs, Float32.(ys))
    else
        xs
    end    
end


(xs, ys) = vectorize(sentences[1], true)
println(size(xs))
println(size(ys))

# 3. Build a RNN model

# input dimension and output dimension
inp = wd + length(partsOfSpeech) + length(chunkTypes) + length(wordShapes)
out = length(entities)

# Create a model
model = Chain(
    GRU(inp, hid),
    Dense(hid, out),
    softmax
)

if (g) model = gpu(model) end

# the full model which takes as input a 2-d matrix representing an input 
# sequence; each column corresponds to a token of the sequence.
function f(x)
    prediction = model(x)
    reset!(model)
    return prediction
end

# 4. Specify a loss function

# Loss function on a batch. For each sequence in the batch, 
# we apply the model and compute the cross-entropy element-wise.
# The total loss of the batch is returned.
loss(xb, yb) = sum(crossentropy.(f.(xb), yb))

# 5. Vectorize the dataset and create batches for training
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

# bring data to GPU if g is true
if g 
    println("Bringing data to GPU...")
    Xb = map(t -> gpu.(t), Xb)
    Yb = map(t -> gpu.(t), Yb)
end
 

# create a data set for training, each training point is a pair of batch
dataset = collect(zip(Xb, Yb))
println("#(batches) = ", length(dataset))

X100 = Xb[100]
Y100 = Yb[100]
println("typeof(X100) = ", typeof(X100)) # this should be Array{Array{Float32,2},1}
println("typeof(Y100) = ", typeof(Y100)) # this should be Array{Array{Float32,2},1}

# 6. Train the model
# train the model with some number of epochs and save the parameters to a BSON file
function train(numEpochs::Int, modelPath::String)
    optimizer = ADAM(.001)
    evalcb = () -> @show(loss(X100, Y100)) # or use loss(dataset[100]...)
    @epochs numEpochs Flux.train!(loss, params(model), dataset, optimizer, cb = throttle(evalcb, 30))
    m = cpu(model)
    @save modelPath m
end

# Predicts the label sequence of a sentence
function predict(sentence::Sentence)::Array{String}
    reset!(model)
    x = vectorize(sentence)
    if (g) x = gpu.(x) end
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

@time train(numEpochs, string(prefix, "dat/ner/vie/vie.bson"))

@time predict(sentences, string(prefix, "dat/ner/vie/vie.train.jul.ner.out"))
@time predict(sentences_test, string(prefix, "dat/ner/vie/vie.test.jul.ner.out"))
