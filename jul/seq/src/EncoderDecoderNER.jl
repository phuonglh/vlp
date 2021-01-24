# phuonglh@gmail.com
# December 2020
# Implementation of a sequence-to-sequence model in Julia

using Flux

using Flux: @epochs
using BSON: @save, @load
using FLoops
using BangBang
using MicroCollections
using StatsBase

include("Sentence.jl")
include("Brick.jl")
include("Embedding.jl")
include("Options.jl")
include("BiRNN.jl")

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
function vocab(sentences::Array{Sentence}, minFreq::Int = 1)::Vocabularies
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
        # crop the columns of xs and ys to maxSequenceLength
        if length(xs) > options[:maxSequenceLength]
            xs = xs[1:options[:maxSequenceLength]]
            ys0 = ys0[1:options[:maxSequenceLength]]
            ys1 = ys1[1:options[:maxSequenceLength]]
        end
        # pad the sequences to the same maximal length if necessary 
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
    Xb = collect(map(A -> collect(A), Iterators.partition(X, options[:batchSize])))
    Yb0 = collect(map(A -> collect(A), Iterators.partition(Y0, options[:batchSize])))
    Yb1 = collect(map(A -> collect(A), Iterators.partition(Y1, options[:batchSize])))
    (Xb, Yb0, Yb1)
end

# For named entity recognition
options = optionsVUD

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
encoder = BiGRU(inputSize, options[:hiddenSize])

"""
    encode(X)

    Encode an index matrix of size (3 x maxSequenceLength) using the embedding and the encoder layers.
"""
encode(X::Array{Int,2}) = encoder(embedding(X))

"""
    encode(Xb)

    Encode a batch, each element in the batch is a matrix representing a sequence, where each column corresponds to 
    a vector representation of a token in the sequence. 
"""
encode(Xb::Array{Array{Int,2},1}) = encode.(Xb)

# 2. Create an attention model which scores the degree of match between 
#  an output position and an input position. The attention model that we use here is simply a linear model.
attention = Dense(2*options[:hiddenSize], 1)

"""
    β(S, H)

    Align a decoder output `s` with hidden states of inputs `h` of size (hiddenSize x m). 
    In the first run, the decoder output only a column vector `s` of length hiddenSize, it should be repeated to create 
    the same number of columns as `h`, that is of size (hiddenSize x m). In subsequent runs, the decoder output should 
    be a matrix of size (hiddenSize x m) and there is no need to repeat columns. We use a broadcast multiplication 
    to combine the two cases into one.

    This function computes attention scores matrix of size (1 x maxSequenceLength) for a decoder position.
"""
function β(S, H::Array{Float32,2})
    V = S .* Float32.(ones(1, size(H,2)))
    attention(vcat(H, V))
end

"""
    α(β)

    Compute the probabilities (weights) vector by using the softmax function. Return a matrix of the same 
    size as β, that is (1 x maxSequenceLength).
"""
function α(β::Array{Float32,2})
    score = exp.(β)
    s = sum(score, dims=2)
    score ./ s
end

# 3. Create a decoder
numLabels = length(labelIndex)
decoder = GRU(options[:hiddenSize] + numLabels, options[:hiddenSize])
linearLayer = Dense(options[:hiddenSize], numLabels)

"""
    decode(H, Y0)

    H is the hidden states of the encoder, which is a matrix of size (hiddenSize x maxSequenceLength)
    and Y0 is an one-hot vector representing a label at position t.
"""
function decode(H::Array{Float32,2}, y0::Array{Int,1})
    w = α(β(decoder.state, H))
    c = sum(w .* H, dims=2)
    v = vcat(y0, c)
    linearLayer(decoder(v))
end

function decode(H::Array{Float32,2}, Y0::Array{Int,2})
    y0s = [Y0[:, t] for t=1:size(Y0,2)]
    ŷs = [decode(H, y0) for y0 in y0s]
    hcat(ŷs...)
end

decode(Hb::Array{Array{Float32,2},1}, Y0b::Array{Array{Int,2},1}) = decode.(Hb, Y0b)

# The full machinary
machine = (embedding, encoder, attention, decoder, linearLayer)

function model(Xb, Y0b)
    Ŷb = decode(encode(Xb), Y0b)
    Flux.reset!(machine)
    return Ŷb
end

"""
    train(options)

"""
function train(options::Dict{Symbol,Any})    
    # create batches of data, each batch
    Xbs, Y0bs, Ybs = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)
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

    # define the loss function
    loss(Xb, Y0b, Yb) = sum(Flux.logitcrossentropy.(model(Xb, Y0b), Yb))

    Ubs, Vbs, Wbs = batch(sentencesValidation, wordIndex, shapeIndex, posIndex, labelIndex)

    optimizer = ADAM(1E-4)
    file = open(options[:logPath], "w")
    write(file, "loss,trainingAccuracy,validationAccuracy\n")
    evalcb = Flux.throttle(30) do
        ℓ = loss(Xbs[1], Y0bs[1], Ybs[1])
        @info string("\tloss = ", ℓ)
        # trainingAccuracy = evaluate(model, Xbs, Y0bs, Ybs)
        # validationAccuracy = evaluate(model, Ubs, Vbs, Wbs)
        # @info string("\tloss = ", ℓ, ", training accuracy = ", trainingAccuracy, ", validation accuracy = ", validationAccuracy)
        # write(file, string(ℓ, ',', trainingAccuracy, ',', validationAccuracy, "\n"))
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
    @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
        Ŷb = Flux.onecold.(model(Xbs[i], Y0bs[i]))
        Yb = Flux.onecold.(Ybs[i])
        # number of tokens and number of matches in this batch
        tokens, matches = 0, 0
        u, v = 0, 0
        for t=1:length(Yb)
            n = options[:maxSequenceLength]
            # find the last position of non-padded element
            while Yb[t][n] == paddingY
                n = n - 1
            end
            tokens += n
            matches += sum(Ŷb[t][1:n] .== Yb[t][1:n])
            js = (Yb[t][1:n] .!= labelIndex["O"])
            u += sum(js)
            v += sum(Ŷb[t][1:n][js] .== Yb[t][1:n][js])
        end
        @reduce(numTokens += tokens, numMatches += matches, numNonOs += u, numNonOMatches += v)
    end
    @info "\tTotal matched tokens = $(numMatches)/$(numTokens)"
    @info "\tTotal non-O matched tokens = $(numNonOMatches)/$(numNonOs)"
    return numMatches/numTokens
end

"""
    predict(model, Xbs, Y0bs, Ybs,  split, paddingY)

    Predict a (training) data set, save result to a CoNLL-2003 evaluation script.
"""
function predict(model, Xbs, Y0bs, Ybs, split::Symbol, paddingY::Int=1)
    numBatches = length(Xbs)
    @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
        Ŷb = Flux.onecold.(model(Xbs[i], Y0bs[i]))
        Yb = Flux.onecold.(Ybs[i])
        truth, pred = Array{Array{String,1},1}(), Array{Array{String,1},1}()
        for t=1:length(Yb)
            n = options[:maxSequenceLength]
            # find the last position of non-padded element
            while Yb[t][n] == paddingY
                n = n - 1
            end
            push!(truth, vocabularies.labels[Yb[t][1:n]])
            push!(pred, vocabularies.labels[Ŷb[t][1:n]])
        end
        @reduce(ss = append!!(EmptyVector(), [(truth, pred)]))
    end
    file = open(options[split], "w")
    result = Array{String,1}()
    for b=1:numBatches
        truths = ss[b][1]
        preds = ss[b][2]
        for i = 1:length(truths)
            x = map((a, b) -> string(a, ' ', b), truths[i], preds[i])
            s = join(x, "\n")
            push!(result, s * "\n")
        end
    end
    write(file, join(result, "\n"))
    close(file)
end


"""
    predict(sentence, labelIndex)

    Find the label sequence for a given sentence.
"""
function predict(sentence, labelIndex::Dict{String,Int})
    Flux.reset!(machine)
    ps = [labelIndex["BOS"]]
    Xs, Y0s, Ys = batch([sentence], wordIndex, shapeIndex, posIndex, labelIndex)
    Xb = first(Xs)
    Hb = encode(Xb)
    Y0 = repeat(Flux.onehotbatch(ps, 1:numLabels), 1,size(Xb[1], 2))
    m = min(length(sentence.tokens), size(Xb[1], 2))
    for t=1:m
        currentY = Flux.onehot(ps[end], 1:numLabels)
        Y0[:,t] = currentY
        Y0b = [ Int.(Y0) ]
        output = decode(Hb, Y0b)
        Ŷ = softmax(output[1][:,t])
        # nextY = Flux.onecold(Ŷ)     # use a hard selection approach, always choose the label with the best probability
        nextY = wsample(1:numLabels, Ŷ) # use a soft selection approach to sample a label from the distribution
        push!(ps, nextY)
    end
    return vocabularies.labels[ps[2:end]]
end

"""
    predict(sentences, labelIndex)
"""
function predict(sentences::Array{Sentence}, labelIndex::Dict{String,Int})
    map(sentence -> predict(sentence, labelIndex), sentences)
end

function diagnose(sentence)
    Xs, Y0s, Ys = batch([sentence], wordIndex, shapeIndex, posIndex, labelIndex)
    Xb = first(Xs)
    H = encode(first(Xb))
    Y0b = first(Y0s)
    Y0 = first(Y0b)
    vocabularies.labels[Flux.onecold(decode(H, Y0))]
end

