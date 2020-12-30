# phuonglh@gmail.com

using Flux
using Flux: @epochs
using BSON: @save, @load
using CUDA

include("EmbeddingWSP.jl")
include("Join.jl")
include("../../tdp/src/Oracle.jl")
include("Options.jl")


struct Vocabularies
    words::Array{String}
    shapes::Array{String}
    partsOfSpeech::Array{String}
    labels::Array{String}
end

function extract(features::Array{String}, prefixes::Array{String})::Array{String}
    xs = filter(feature -> startswith(feature, prefixes[1]) || startswith(feature, prefixes[2]), features)
    ws = map(x -> x[findfirst(':', x)+1:end], xs)
    if length(ws) < options[:featuresPerContext]
        for _=1:(options[:featuresPerContext] - length(ws))
            append!(ws, [options[:padding]])
        end
    end
    @assert length(ws) == options[:featuresPerContext]
    return ws
end

"""
    vocab(contexts, minFreq)

    Builds vocabularies of words and transitions. The word vocabulary is sorted by frequency.
    Only features whose count is greater than `minFreq` are kept.
"""    
function vocab(contexts::Array{Context}, minFreq::Int = 2)::Vocabularies
    transitions = map(context -> context.transition, contexts)
    words = Iterators.flatten(map(context -> extract(context.features, ["ws", "wq"]), contexts))
    wordFrequency = Flux.frequencies(map(lowercase, words))
    filter!(p -> p.second >= minFreq, wordFrequency)
    shapes = Iterators.flatten(map(context -> extract(context.features, ["ss", "sq"]), contexts))
    partsOfSpeech = Iterators.flatten(map(context -> extract(context.features, ["ts", "tq"]), contexts))
    Vocabularies(collect(keys(wordFrequency)), unique(shapes), unique(partsOfSpeech), unique(transitions))
end


"""
    vectorize(sentence, wordIndex, shapeIndex, posIndex, labelIndex)

    Vectorize a training sentence. An oracle is used to extract (context, transition) pairs from 
    the sentence. Then each context is vectorized to a tuple of (token matrix of the sentence, word id array of the context).
    The word id array of the sentence is the same across all contexts. This function returns an array of pairs (xs, ys) where 
    each xs is a pair (ws, x). Each token matrix is a 3-row matrix corresponding to the word id, shape id, and part-of-speech id arrays.
"""
function vectorize(sentence::Sentence, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int})::Array{Tuple{Tuple{Array{Int},Array{Int}},Int}}
    ws = map(token -> [get(wordIndex, lowercase(token.word), 1), shapeIndex[shape(token.word)], posIndex[token.annotation[:pos]]], sentence.tokens)
    if length(ws) < options[:maxSequenceLength]
        for _ = 1:(options[:maxSequenceLength]-length(ws))
            append!(ws, [[wordIndex[options[:padding]], 1, 1]]) # use 1 as padding for shape and part-of-speech
        end
    else
        ws = ws[1:options[:maxSequenceLength]]
    end
    append!(ws, [[wordIndex[options[:padding]], 1, 1]]) # add the last padding token
    contexts = decode(sentence)
    fs = map(context -> extract(context.features, ["ws", "wq"]), contexts)
    words = map(token -> lowercase(token.word), sentence.tokens)
    append!(words, [options[:padding]])
    positionIndex = Dict{String,Int}(word => i for (i, word) in enumerate(words))
    xs = map(f -> map(word -> positionIndex[lowercase(word)], f), fs)
    ys = map(context -> labelIndex[context.transition], contexts)
    # return a collection of tuples for this sentence, use Flux.batch to convert ws to a matrix of size 3x(maxSequenceLength+1).
    collect(zip(map(x -> (Flux.batch(ws), x), xs), ys))
end

"""
    batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)

    Create batches of data for training or evaluating. Each batch contains a pair (`Xb`, `Yb`) where 
    `Xb` is an array of `batchSize` samples. `Yb` is an one-hot matrix of size (`numLabels` x `batchSize`).
"""
function batch(sentences::Array{Sentence}, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int})
    # vectorizes all sentences and flatten the samples
    samples = collect(Iterators.flatten(map(sentence -> vectorize(sentence, wordIndex, shapeIndex, posIndex, labelIndex), sentences)))
    X = map(sample -> sample[1], samples)   
    y = map(sample -> sample[2], samples)
    # build batches of data for training
    Xs = collect(Iterators.partition(X, options[:batchSize]))
    Y = collect(Iterators.partition(y, options[:batchSize]))
    # convert each output batch to an one-hot matrix
    Ys = map(b -> Flux.onehotbatch(b, 1:length(labelIndex)), Y)
    (Xs, Ys)
end

"""
    evaluate(mlp, Xs, Ys)

    Evaluate the accuracy of a model on a dataset.
"""
function evaluate(mlp, Xs, Ys)
    Ŷb = Flux.onecold.(mlp.(Xs)) |> cpu
    Yb = Flux.onecold.(Ys) |> cpu
    pairs = collect(zip(Ŷb, Yb))
    matches = map(p -> sum(p[1] .== p[2]), pairs)
    numSamples = sum(map(y -> length(y), Yb))
    accuracy = reduce((a, b) -> a + b, matches)/numSamples
    return accuracy
end


"""
    train(options)

    Train a classifier model.
"""
function train(options)
    sentences = readCorpus(options[:testCorpus])
    contexts = collect(Iterators.flatten(map(sentence -> decode(sentence), sentences)))
    @info "#(contexts) = $(length(contexts))"
    vocabularies = vocab(contexts)
    prepend!(vocabularies.words, [options[:unknown]])
    append!(vocabularies.words, [options[:padding]])
    labelIndex = Dict{String, Int}(label => i for (i, label) in enumerate(vocabularies.labels))
    wordIndex = Dict{String, Int}(word => i for (i, word) in enumerate(vocabularies.words))
    shapeIndex = Dict{String, Int}(shape => i for (i, shape) in enumerate(vocabularies.shapes))
    posIndex = Dict{String, Int}(tag => i for (i, tag) in enumerate(vocabularies.partsOfSpeech))

    vocabSize = min(options[:vocabSize], length(wordIndex))

    # build training dataset
    Xs, Ys = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)
    dataset = collect(zip(Xs, Ys))
    @info "numBatches  = $(length(dataset))"    
    @info "vocabSize = $(vocabSize)"

    mlp = Chain(
        Join(
            EmbeddingWSP(vocabSize, options[:wordSize], length(shapeIndex), options[:shapeSize], length(posIndex), options[:posSize]),
            GRU(options[:wordSize] + options[:shapeSize] + options[:posSize], options[:hiddenSize]),
        ),
        Dense(options[:featuresPerContext] * options[:hiddenSize], length(labelIndex))
    )

    # save the vocabulary and label index to external files
    file = open(options[:wordPath], "w")
    for f in vocabularies.words
        write(file, string(f, " ", wordIndex[f]), "\n")
    end
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

    # bring the dataset and the model to GPU if any
    if options[:gpu]
        @info "Bring data to GPU..."
        dataset = map(p -> p |> gpu, dataset)
        mlp = mlp |> gpu
    end
    @info typeof(dataset[1][1]), size(dataset[1][1])
    @info typeof(dataset[1][2]), size(dataset[1][2])

    @info "Total weight of initial word embeddings = $(sum(mlp[1].fs[1].word.W))"

    # define a loss function, an optimizer and train the model
    loss(x, y) = Flux.logitcrossentropy(hcat(mlp.(x)...), y)
    optimizer = ADAM()
    file = open(options[:logPath], "w")
    evalcb = Flux.throttle(30) do
        ℓ = loss(dataset[1]...)
        accuracy = evaluate(mlp, Xs, Ys)
        @info string("loss = $(ℓ), accuracy = $(accuracy)")
        write(file, string(ℓ, ',', accuracy, "\n"))
    end
    # train the model
    @time @epochs options[:numEpochs] Flux.train!(loss, params(mlp), dataset, optimizer, cb = evalcb)
    close(file)
    @info "Total weight of final word embeddings = $(sum(mlp[1].fs[1].word.W))"

    # evaluate the model on the training set
    @info "Evaluating the model..."
    accuracy = evaluate(mlp, Xs, Ys)
    @info "Training accuracy = $accuracy"
    # save the model to a BSON file
    mlp = mlp |> cpu
    @save options[:modelPath] mlp
    mlp    
end

