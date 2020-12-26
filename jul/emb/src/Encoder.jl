# phuonglh@gmail.com
# Sentence encoder which encodes a sequence of tokens into a sequence of 
# dense vectors. 

using Flux
using Flux: @epochs
using BSON: @save, @load

using FLoops

include("Embedding.jl")
include("Options.jl")
include("../../tdp/src/Sentence.jl")
include("../../tok/src/Brick.jl")


struct Vocabularies
    words::Array{String}
    shapes::Array{String}
    partsOfSpeech::Array{String}
    labels::Array{String}
end

"""
    vocab(contexts, minFreq)

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
        partsOfSpeech[token.annotation[:upos]] = 0
        labels[token.annotation[:head]] = 0
    end
    # filter out infrequent words
    filter!(p -> p.second >= minFreq, wordFrequency)
    
    Vocabularies(collect(keys(wordFrequency)), collect(keys(shapes)), collect(keys(partsOfSpeech)), collect(keys(labels)))
end

"""
    batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)

    Create batches of data for training or evaluating. Each batch contains a pair (Xb, Yb) where 
    Xb is an array of matrices of size (featuresPerToken x maxSequenceLength). Each column of Xb is a vector representing a token.
    If a sentence is shorter than maxSequenceLength, it is padded with vectors of ones.
"""
function batch(sentences::Array{Sentence}, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int})
    X, Y = Array{Array{Int,2},1}(), Array{Array{Int,2},1}()
    for sentence in sentences
        xs = map(token -> [get(wordIndex, lowercase(token.word), 1), shapeIndex[shape(token.word)], posIndex[token.annotation[:upos]]], sentence.tokens)
        ys = map(token -> Flux.onehot(labelIndex[token.annotation[:head]], 1:length(labelIndex), 1), sentence.tokens)
        # pad the columns of xs and ys to maxSequenceLength
        if length(xs) > options[:maxSequenceLength]
            xs = xs[1:options[:maxSequenceLength]]
            ys = ys[1:options[:maxSequenceLength]]
        end
        for t=length(xs)+1:options[:maxSequenceLength]
            push!(xs, ones(3))
            push!(ys, Flux.onehot(1, 1:length(labelIndex)))
        end
        push!(X, Flux.batch(xs))
        push!(Y, Flux.batch(ys))
    end
    # build batches of data for training
    Xb = Iterators.partition(X, options[:batchSize])
    Yb = Iterators.partition(Y, options[:batchSize])
    # stack each input batch to a 3-d matrix
    Xs = map(b -> Int.(Flux.batch(b)), Xb)
    # stack each output batch to a 3-d matrix
    Ys = map(b -> Int.(Flux.batch(b)), Yb)
    (Xs, Ys)
end



"""
    train(options)

    Train an encoder.
"""
function train(options::Dict{Symbol,Any})
    sentences = readCorpus(options[:trainCorpus])
    vocabularies = vocab(sentences)
    
    prepend!(vocabularies.words, ["UNK"])
    wordIndex = Dict{String,Int}(word => i for (i, word) in enumerate(vocabularies.words))
    shapeIndex = Dict{String,Int}(shape => i for (i, shape) in enumerate(vocabularies.shapes))
    posIndex = Dict{String,Int}(pos => i for (i, pos) in enumerate(vocabularies.partsOfSpeech))
    labelIndex = Dict{String,Int}(label => i for (i, label) in enumerate(vocabularies.labels))
    
    # create batches of data, each batch is a 3-d matrix of size 3 x maxSequenceLength x batchSize
    Xs, Ys = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)
    dataset = collect(zip(Xs, Ys))
    @info "vocabSize = ", length(wordIndex)
    @info "shapeSize = ", length(shapeIndex)
    @info "posSize = ", length(posIndex)
    @info "numLabels = ", length(labelIndex)
    @info "numBatches  = ", length(dataset)
    @info size(Xs[1])
    @info size(Ys[1])

    # define a model for sentence encoding
    encoder = Chain(
        EmbeddingWSP(min(length(wordIndex), options[:vocabSize]), options[:wordSize], length(shapeIndex), options[:shapeSize], length(posIndex), options[:posSize]),
        GRU(options[:wordSize] + options[:shapeSize] + options[:posSize], options[:hiddenSize]),
        Dense(options[:hiddenSize], length(labelIndex), relu)
    )

    @info "Total weights of initial word embeddings = $(sum(encoder[1].word.W))"

    """
        loss(X, Y)

        Compute the loss on one batch of data where X and Y are 3-d matrices of size (K x maxSequenceLength x batchSize).
        We use the log cross-entropy loss to measure the average distance between prediction and true sequence pairs.
    """
    function loss(X, Y)
        b = size(X,3)
        predictions = [encoder(X[:,:,i]) for i=1:b]
        truths = [Y[:,:,i] for i=1:b]
        sum(Flux.logitcrossentropy(predictions[i], truths[i]) for i=1:b)
    end

    optimizer = ADAM()
    file = open(options[:logPath], "w")
    evalcb = Flux.throttle(30) do
        ℓ = loss(dataset[1]...)
        @info string("loss = ", ℓ)
        write(file, string(ℓ, "\n"))
    end
    # train the model
    @time @epochs options[:numEpochs] Flux.train!(loss, params(encoder), dataset, optimizer, cb = evalcb)
    close(file)
    @info "Total weights of final word embeddings = $(sum(encoder[1].word.W))"
    @info "Evaluating the model on the training set..."
    accuracy = evaluate(encoder, Xs, Ys)
    @info "Training accuracy = $accuracy"
    # save the model to a BSON file
    @save options[:modelPath] encoder
    # save the vocabulary, shape, part-of-speech and label information into external files
    file = open(options[:vocabPath], "w")
    for f in vocabularies.words
        write(file, string(f, " ", wordIndex[f]), "\n")
    end
    close(file)
    file = open(options[:shapePath], "w")
    for f in vocabularies.shapes
        write(file, string(f, " ", shapeIndex[f]), "\n")
    end
    close(file)
    file = open(options[:posPath], "w")
    for f in vocabularies.partsOfSpeech
        write(file, string(f, " ", posIndex[f]), "\n")
    end
    close(file)
    file = open(options[:labelPath], "w")
    for f in vocabularies.labels
        write(file, string(f, " ", labelIndex[f]), "\n")
    end
    close(file)

    encoder
end

"""
    evaluate(encoder, Xs, Ys)

    Evaluate the accuracy of the encoder on a dataset. `Xs` is a list of 3-d input matrices and `Ys` is a list of 
    3-d ground-truth output matrices. 
"""
function evaluate(encoder, Xs, Ys)
    b = options[:batchSize]
    numTokens = reduce((a, b) -> a + b, map(X -> size(X,2) * size(X,3), Xs))
    @floop ThreadedEx(basesize=b÷options[:numCores]) for i=1:b
        Ŷb = [Flux.onecold.(encoder(Xs[i][:,:,t]) for t=1:options[:batchSize])]
        Yb = [Flux.onecold.(Ys[i][:,:,t] for t=1:options[:batchSize])]
        pairs = collect(zip(Ŷb, Yb))
        matches = sum(map(p -> sum(p[1] .== p[2]), pairs))
        @info matches
        @reduce(s += matches)
    end
    @info s
    return s/numTokens
end

encoder = train(options)
