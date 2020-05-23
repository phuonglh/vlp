"""
    RNN-based part-of-speech tagger.
    phuonglh
    January 15, 2020
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
include("BiRNN.jl")
include("Split.jl")

# Some constants
prefix = string(homedir(), "/vlp/")
minFreq = 1
maxSequenceLength = 40
wd = 50
makeLowercase = true
numEpochs = 20
hid = 128


# load pre-trained word vectors
@time wordVectors = load("/opt/data/emb/skip.vie.50d.txt")


# Read part-of-speech tagged sentences
corpus = readVLSP(string(prefix, "dat/tag/vtb-tagged.txt"))
println("#(corpus) = ",length(corpus))
# filter sentences of length not greater than maxSequenceLength
sentences = filter(s -> length(s.tokens) <= maxSequenceLength, corpus)
println("#(sentences of length <= $(maxSequenceLength)) = ", length(sentences))
# build a character index (char => index) and then get the alphabet
charIndex = Dict{Char, Int}()
for sentence in sentences
    words = map(token -> token.text, sentence.tokens)
    for word in words
        for c in word
            if (!haskey(charIndex, c))
                charIndex[c] = length(charIndex)
            end
        end
    end
end
alphabet = collect(keys(charIndex))
println("alphabet = ", alphabet)
println("#(alphabet) = ", length(alphabet))

"""
    Bag-of-character vectors for s[2:end-1]
"""
function boc(s, alphabet)
    a = onehotbatch(collect(s[nextind(s, 1):prevind(s, lastindex(s))]), alphabet)
    # sum the columns
    b = zeros(length(alphabet))
    for j = 1:size(a, 2)
        b = b + a[:,j]
    end
    b
end

function train(sentences_train::Array{Sentence}, sentences_test::Array{Sentence}, hid::Int = 128)
    # 1. build vocab
    wordList = vocab(sentences_train, minFreq, makeLowercase)
    push!(wordList, "<number>")
    push!(wordList, "UNK")
    println("#(vocab) = ", length(wordList))
    # 2. prepare the word embedding matrix and an embedding layer
    N = length(wordList)
    W = rand(wd, N)
    for i = 1:N
        word = wordList[i]
        if (haskey(wordVectors, word))
            W[:, i] = wordVectors[word]
        end
    end
    embed = Embedding(W)
    # 3. build a word index (word => index) of lower-case words
    wordIndex = Dict{String, Int}(word => i for (i, word) in enumerate(wordList))
    partsOfSpeech = labels(sentences_train, 'p')
    println("labels = ", partsOfSpeech)
    wordShapes = labels(sentences, 's')
    println("word shapes = ", wordShapes)
    
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
        # one-hot shape vector
        shs = map(token -> token.properties['s'], tokens)
        ss = onehotbatch(shs, wordShapes)
        # one-hot first char vector
        ucs = map(token -> first(token.text), tokens)
        us = onehotbatch(ucs, alphabet)
        # one-hot last char vector
        vcs = map(token -> last(token.text), tokens)
        vs = onehotbatch(vcs, alphabet)
        # one-hot middle bag-of-character vector
        cs = zeros(length(alphabet), length(tokens))
        for j = 1:length(tokens)
            cs[:,j] = boc(tokens[j].text, alphabet)
        end
        # combine all vectors into xs and convert xs to Float32 to speed up computation
        xs = Float32.(vcat(embed(ws), ss, us, vs, cs))
        #xs = Float32.(embed(ws)) # if do not use shapes, the accuracy decreases about 3%
        if (training)
            yy = map(token -> token.properties['p'], tokens)
            ys = onehotbatch(yy, partsOfSpeech)
            (xs, Float32.(ys))
        else
            xs
        end    
    end
    (xs, ys) = vectorize(sentences_train[1], true)
    println(size(xs))
    println(size(ys))
    
    # 4. create a model
    # input dimension and output dimension
    inp = wd + length(wordShapes) + 3*length(alphabet)
    out = length(partsOfSpeech)
    # project the input embedding to some dimension
    dim = 256

    # Create a model
    model = Chain(
        Dense(inp, dim, relu),
        GRU(dim, hid),
        Dense(hid, out),
        softmax
    )
    println(model)
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

    batchSize = 32
    println("Vectorizing the dataset... Please wait.")
    # XYs is an array of samples  [(x_1, y_1), (x_2, y_2,),... ]
    @time XYs = map(s -> vectorize(s, true), sentences_train)
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
        map(e -> partsOfSpeech[e], y)
    end

    # Predicts a list of sentences, collect prediction result and report prediction accuracy
    @sync function predict(sentences::Array{Sentence})::Tuple{Float64,Array{Array{String}}}
        zs = pmap(s -> predict(s), sentences)
        ys = map(s -> map(token -> token.properties['p'], s.tokens), sentences)
        total = 0
        correct = 0
        for i = 1:length(ys)
            (y, z) = (ys[i], zs[i])
            total = total + length(y)
            correct = correct + sum(y .== z)
        end
        ts = map(p -> map(q -> string(q[1], '/', q[2]), zip(p[1], p[2])), zip(zs, ys))
        (100 * correct/total, ts)
    end

    @time train(numEpochs, string(prefix, "dat/tag/vie.bson"))

    (train_score, train_prediction) = predict(sentences_train)
    (test_score, test_prediction) = predict(sentences_test)
    
    # write the prediction to output files
    file = open(string(prefix, "dat/tag/train.prediction.txt"), "w")
    for s in train_prediction
        write(file, join(s, " "))
        write(file, "\n")
    end
    close(file)
    file = open(string(prefix, "dat/tag/test.prediction.txt"), "w")
    for s in test_prediction
        write(file, join(s, " "))
        write(file, "\n")
    end
    close(file)
    # confusion matrix
#    A = zeros(out, out)
    # build a tag index
#    tagIndex = Dict{String, Int}(tag => i for (i, tag) in enumerate(partsOfSpeech))

    # return a pair of accuracy scores
    (train_score, test_score)
end

# split test/training datasets and log accuracy scores to an external file
pairs = folds(5, collect(1:length(sentences)))
# run 5 models
output = string(prefix, "dat/tag/rnn.pos.txt")
file = open(output, append=true)

for p in pairs
    sentences_test = sentences[p[1]]
    sentences_train = sentences[p[2]]
    train_accuracy, test_accuracy = train(sentences_train, sentences_test, hid)
    println("train accuracy = ", train_accuracy, ", test accuracy = ", test_accuracy)
    write(file, string("hid = ", hid, "\t", train_accuracy, "\t", test_accuracy, "\n"))
    flush(file)
end
close(file)
