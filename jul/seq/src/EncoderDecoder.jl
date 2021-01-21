# phuonglh@gmail.com
# December 2020
# Implementation of a sequence-to-sequence model in Julia

using Flux

using Flux: @epochs
using BSON: @save, @load
using FLoops
using BangBang
using MicroCollections

include("Sentence.jl")
include("Brick.jl")
include("Embedding.jl")
include("Options.jl")

struct Vocabularies
    words::Array{String}
    shapes::Array{String}
    partsOfSpeech::Array{String}
    labels::Array{String}
end

"""
    vocab(sentences, minFreq)

    Builds vocabularies of words, shapes, parts-of-speech, and labels. The word vocabulary is sorted by frequency.
    Only words whose count is greater than `minFreq` are kept.
"""    
function vocab(sentences::Array{Sentence}, minFreq::Int = 2)::Vocabularies
    tokens = Iterators.flatten(map(sentence -> sentence.tokens, sentences))
    wordFrequency = Dict{String, Int}()
    shapes = Dict{String,Int}()
    partsOfSpeech = Dict{String,Int}()
    labels = Dict{String,Int}()
    for token in tokens
        word = lowercase(strip(token.word))
        haskey(wordFrequency, word) ? wordFrequency[word] += 1 : wordFrequency[word] = 1
        shapes[shape(token.word)] = 0
        partsOfSpeech[token.annotation[:p]] = 0
        labels[token.annotation[:e]] = 0
    end
    # filter out infrequent words
    filter!(p -> p.second >= minFreq, wordFrequency)
    
    Vocabularies(collect(keys(wordFrequency)), collect(keys(shapes)), collect(keys(partsOfSpeech)), collect(keys(labels)))
end

"""
    batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex, options)

    Create batches of data for training or evaluating. Each batch contains a triple (Xb, Yb0, Yb) where 
    Xb is an array of matrices of size (featuresPerToken x maxSequenceLength). Each column of Xb is a vector representing a token.
    If a sentence is shorter than maxSequenceLength, it is padded with vectors of ones.
"""
function batch(sentences::Array{Sentence}, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int}, options=optionsVLSP2016)
    X, Y0, Y1 = Array{Array{Int,2},1}(), Array{Array{Int,2},1}(), Array{Array{Int,2},1}()
    paddingX = [wordIndex[options[:paddingX]]; 1; 1]
    numLabels = length(labelIndex)
    paddingY = Flux.onehot(labelIndex[options[:paddingY]], 1:numLabels)
    for sentence in sentences
        xs = map(token -> [get(wordIndex, lowercase(token.word), 1), get(shapeIndex, shape(token.word), 1), get(posIndex, token.annotation[:p], 1)], sentence.tokens)
        push!(xs, paddingX)
        ys = map(token -> Flux.onehot(labelIndex[token.annotation[:e]], 1:numLabels, 1), sentence.tokens)
        ys0 = copy(ys); prepend!(ys0, [Flux.onehot(labelIndex["BOS"], 1:length(labelIndex), 1)])
        ys1 = copy(ys); append!(ys1, [Flux.onehot(labelIndex["EOS"], 1:length(labelIndex), 1)])
        # pad the columns of xs and ys to maxSequenceLength
        if length(xs) > options[:maxSequenceLength]
            xs = xs[1:options[:maxSequenceLength]]
            ys0 = ys0[1:options[:maxSequenceLength]]
            ys1 = ys1[1:options[:maxSequenceLength]]
        end
        # pad the sequences to the same maximal length
        for t=length(xs)+1:options[:maxSequenceLength]
            push!(xs, paddingX) 
            push!(ys0, paddingY)
            push!(ys1, paddingY)
        end
        push!(X, Flux.batch(xs))
        push!(Y0, Flux.batch(ys0))
        push!(Y1, Flux.batch(ys1))
    end
    # build batches of data for training
    Xb = Iterators.partition(X, options[:batchSize])
    Yb0 = Iterators.partition(Y0, options[:batchSize])
    Yb1 = Iterators.partition(Y1, options[:batchSize])
    (Xb, Yb0, Yb1)
end

# For named entity recognition
options = optionsVLSP2016

# Read training data and create indices
sentences = readCorpusCoNLL(options[:trainCorpus])
sentencesValidation = readCorpusCoNLL(options[:validCorpus])
@info "Number of training sentences = $(length(sentences))"
@info "Number of validation sentences = $(length(sentencesValidation))"
vocabularies = vocab(sentences)

prepend!(vocabularies.words, [options[:unknown]])
append!(vocabularies.words, [options[:paddingX]])
wordIndex = Dict{String,Int}(word => i for (i, word) in enumerate(vocabularies.words))
shapeIndex = Dict{String,Int}(shape => i for (i, shape) in enumerate(vocabularies.shapes))
posIndex = Dict{String,Int}(pos => i for (i, pos) in enumerate(vocabularies.partsOfSpeech))
prepend!(vocabularies.labels, [options[:paddingY]])
append!(vocabularies.labels, ["BOS", "EOS"])
labelIndex = Dict{String,Int}(label => i for (i, label) in enumerate(vocabularies.labels))

"""
    saveIndex(index, path)
    
    Save an index to an external file.
"""
function saveIndex(index, path)
    file = open(path, "w")
    for f in keys(index)
        write(file, string(f, " ", index[f]), "\n")
    end
    close(file)
end

"""
    loadIndex(path)

    Load an index from a file which is previously saved by `saveIndex()` function.
"""
function loadIndex(path)::Dict{String,Int}
    lines = readlines(path)
    pairs = Array{Tuple{String,Int},1}()
    for line in lines
        parts = split(line, " ")
        push!(pairs, (string(parts[1]), parse(Int, parts[2])))
    end
    return Dict(pair[1] => pair[2] for pair in pairs)
end

# 0. Create an embedding layer
embedding = EmbeddingWSP(
    min(length(wordIndex), options[:vocabSize]), options[:wordSize], 
    length(shapeIndex), options[:shapeSize], 
    length(posIndex), options[:posSize]
)
inputSize = options[:wordSize] + options[:shapeSize] + options[:posSize]

# 1. Create an encoder
forwardEncoder = GRU(inputSize, options[:hiddenSize]÷2)
backwardEncoder = GRU(inputSize, options[:hiddenSize]÷2)

"""
    encode(Xb)

    Encode a batch, each element in the batch is a matrix representing a sequence, where each column corresponds to 
    a vector representation of a token in the sequence. We use a bi-directional GRU for to encode each input matrix
    of the batch. Xb is of type Array{Array{Int,2},1}.
"""
function encode(Xb)
    Hb = embedding.(collect(Xb))
    vcat.(forwardEncoder.(Hb), Flux.flip(backwardEncoder, Hb))
end

# 2. Create an attention model which scores the degree of match between 
#  an output position and an input position. The attention model that we use here is simply a linear model.
attention = Dense(2*options[:hiddenSize], 1)

"""
    β(s, h)

    Align a decoder output `s` with hidden states of inputs `h` of size (hiddenSize x m). 
    In the first run, the decoder output only a column vector `s` of length hiddenSize, it should be repeated to create 
    the same number of columns as `h`, that is of size (hiddenSize x m). In subsequent runs, the decoder output should 
    be a matrix of size (hidden x m) and there is no need to repeat columns.

    We then take each decoder output vector at position s_t 
    and concatenate it with hidden states s_j of the encoder, for all j = 1,2,...,m, and compute attention scores for position t. 

    Since there are n positions, this function returns a score matrix of size n x m.
"""
function β(s, h::Array{Float32,2})
    vv = if length(size(s)) == 1
        repeat(s, 1, size(h, 2))
    else 
        s
    end
    vs = [vcat(h, repeat(vv[:, t], 1, size(h,2))) for t = 1:size(vv, 2)]
    bs = attention.(vs)
    vcat(bs...)
end

"""
    α(βb)

    Compute the weights α for a batch of samples, each sample is a 2-d matrix of size (n x m)
    representing attention score matrices.
"""
function α(βb::Array{Array{Float32,2},1})
    βs = map(β -> exp.(β), βb)
    ss = map(β -> sum(β, dims=2), βs)
    map((β, s) -> β./s, βs, ss)
end

# 3. Create a decoder
numLabels = length(labelIndex)
decoder = LSTM(options[:hiddenSize] + numLabels, options[:hiddenSize])
linearLayer = Dense(options[:hiddenSize], numLabels)

"""
    decode(Hb, Yb)

    Decode a batch.
"""
function decode(Hb, Yb)
    Wb = α([β(decoder.state[2], h) for h in Hb])
    # compute a weighted sum; note that we need to transform a vector to a row vector before multiplying to a matrix 
    # for a correct broadcast of dimension
    weightedSum(w, h) = [sum(w[t,:]' .* h, dims=2) for t=1:size(w,1)]
    Cb = map((w, h) -> weightedSum(w, h), Wb, Hb) 
    # each element in Cb is an array of matrices of size (hiddenState x 1), using hcat, we create a matrix of size (hiddenState x n)
    contexts = map(cs -> hcat(cs...), Cb) 
    Ub = map((y, context) -> vcat(Float32.(y), context), Yb, contexts)
    Vb = linearLayer.(decoder.(Ub))
    return softmax.(Vb)
end

# The full machinary
machine = Chain(embedding, forwardEncoder, backwardEncoder, attention, decoder, linearLayer)

"""
    model(Xb, Yb)
"""
function model(Xb, Yb)
    prediction = decode(encode(Xb), Yb)
    Flux.reset!(machine)
    return prediction
end

"""
    train(options)

"""
function train(options::Dict{Symbol,Any})    
    # create batches of data, each batch is a 3-d matrix of size 3 x maxSequenceLength x batchSize
    Xs, Ys0, Ys = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)
    Xbs, Y0bs, Ybs = collect(Xs), collect(Ys0), collect(Ys)
    dataset = collect(zip(Xbs, Y0bs, Ybs))

    @info "vocabSize = ", length(wordIndex)
    @info "shapeSize = ", length(shapeIndex)
    @info "posSize = ", length(posIndex)
    @info "numLabels = ", length(labelIndex)
    @info "numBatches  = ", length(dataset)

    # save the vocabulary, shape, part-of-speech and label information to external files
    saveIndex(wordIndex, options[:wordPath])
    saveIndex(shapeIndex, options[:shapePath])
    saveIndex(posIndex, options[:posPath])
    saveIndex(labelIndex, options[:labelPath])

    @info "Total weight of inial word embeddings = $(sum(embedding.word.W))"

    function loss(Xb, Y0b, Yb)
        sum(Flux.crossentropy.(model(Xb, Y0b), Yb))
    end

    Us, Vs, Ws = batch(sentencesValidation, wordIndex, shapeIndex, posIndex, labelIndex)
    Ubs, Vbs, Wbs = collect(Us), collect(Vs), collect(Ws)

    optimizer = ADAM()
    file = open(options[:logPath], "w")
    write(file, "loss,trainingAccuracy,validationAccuracy\n")
    evalcb = Flux.throttle(30) do
        ℓ = loss(Xbs[1], Y0bs[1], Ybs[1])
        trainingAccuracy = evaluate(model, Xbs, Y0bs, Ybs)
        validationAccuracy = evaluate(model, Ubs, Vbs, Wbs)
        @info string("\tloss = ", ℓ, ", training accuracy = ", trainingAccuracy, ", validation accuracy = ", validationAccuracy)
        write(file, string(ℓ, ',', trainingAccuracy, ',', validationAccuracy, "\n"))
    end
    
    # train the model
    @time @epochs options[:numEpochs] Flux.train!(loss, params(machine), dataset, optimizer, cb = evalcb)
    close(file)
    # save the model to a BSON file
    @save options[:modelPath] machine

    @info "Total weight of final word embeddings = $(sum(embedding.word.W))"
    @info "Evaluating the model on the training set..."
    accuracy = evaluate(model, Xbs, Y0bs, Ybs)
    @info "Training accuracy = $accuracy"
    accuracyValidation = evaluate(model, Ubs, Vbs, Wbs)
    @info "Validation accuracy = $accuracyValidation"
    machine
end

"""
    evaluate(model, Xbs, Y0bs, Ybs, paddingY)

    Evaluate the accuracy of the model on a dataset. 
"""
function evaluate(model, Xbs, Y0bs, Ybs, paddingY::Int=1)
    numBatches = length(Xbs)
    # normally, size(X,3) is the batch size except the last batch
    @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
        Ŷb = Flux.onecold.(model(Xbs[i], Y0bs[i]))
        Yb = Flux.onecold.(Ybs[i])
        # number of tokens and number of matches in this batch
        tokens, matches = 0, 0
        for t=1:length(Yb)
            n = options[:maxSequenceLength]
            # find the last position of non-padded element
            while Yb[t][n] == paddingY
                n = n - 1
            end
            tokens += n
            matches += sum(Ŷb[t][1:n] .== Yb[t][1:n])
        end
        @reduce(numTokens += tokens, numMatches += matches)
    end
    @info "\tTotal matched tokens = $(numMatches)/$(numTokens)"
    return numMatches/numTokens
end

"""
    predict(sentence, labelIndex)

    Find the label sequence for a given sentence.
"""
function predict(sentence, labelIndex::Dict{String,Int})
    labels = fill("", length(labelIndex))
    for key in keys(labelIndex)
        labels[labelIndex[key]] = key
    end
    ps = [labelIndex["BOS"]]
    numLabels = length(labels)
    Xs, Ys0, Ys = batch([sentence], wordIndex, shapeIndex, posIndex, labelIndex)  
    Xb = collect(first(Xs))
    m = size(Xb[1], 2)
    Y = repeat(Flux.onehotbatch(ps, 1:numLabels), 1, m)
    Yb = [Y]
    for t=1:m
        nextY = Flux.onehot(ps[end], 1:numLabels)
        Y[:,t] = nextY
        Yb = [ Y ]
        output = model(Xb, Yb)
        Ŷ = output[1][:,t]
        next = Flux.onecold(Ŷ)
        push!(ps, next)
    end
    return labels[ps[2:end]]
end
