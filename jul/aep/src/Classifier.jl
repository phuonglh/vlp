# phuonglh@gmail.com, December 2020.

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

"""
    extract(features, prefixes)

    Extract selected features from an array of features obtained by the Oracle.jl module.
"""
function extract(features::Array{String}, prefixes::Array{String})::Array{String}
    xs = filter(feature -> startswith(feature, prefixes[1]) || startswith(feature, prefixes[2]), features)
    ws = map(x -> x[findfirst(':', x)+1:end], xs)
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
    vectorizeSentence(sentence, wordIndex, shapeIndex, posIndex)

    Vectorize a given input sentence into an integer matrix of size 3 x maxSentenceLength. The first row contains word indices,
    the second row contains shape indices, and the third row contains part-of-speech indices.
"""
function vectorizeSentence(sentence::Sentence, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int})
    function featurize(token)
        w = get(wordIndex, lowercase(token.word), wordIndex[options[:unknown]])
        s = get(shapeIndex, shape(token.word), shapeIndex[options[:padding]])
        t = get(posIndex, token.annotation[:pos], posIndex[options[:padding]])
        return [w, s, t]
    end
    ws = map(token -> featurize(token), sentence.tokens)
    pad = options[:padding]
    if length(ws) < options[:maxSequenceLength]
        for _ = 1:(options[:maxSequenceLength]-length(ws))
            append!(ws, [[wordIndex[pad], shapeIndex[pad], posIndex[pad]]])
        end
    else
        ws = ws[1:options[:maxSequenceLength]]
    end
    append!(ws, [[wordIndex[pad], shapeIndex[pad], posIndex[pad]]]) # add the last padding token
    return ws
end

"""
    vectorize(sentence, wordIndex, shapeIndex, posIndex, labelIndex)

    Vectorize a training sentence. An oracle is used to extract (context, transition) pairs from 
    the sentence. Then each context is vectorized to a tuple of (token matrix of the sentence, word id array of the context).
    The word id array of the sentence is the same across all contexts. This function returns an array of pairs (xs, ys) where 
    each xs is a pair (ws, x). Each token matrix is a 3-row matrix corresponding to the word id, shape id, and part-of-speech id arrays.
"""
function vectorize(sentence::Sentence, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int})
    ws = vectorizeSentence(sentence, wordIndex, shapeIndex, posIndex)
    contexts = decode(sentence)
    fs = map(context -> extract(context.features, ["ws", "wq"]), contexts)
    words = map(token -> lowercase(token.word), sentence.tokens)
    append!(words, [options[:padding]])
    positionIndex = Dict{String,Int}(word => i for (i, word) in enumerate(words))
    xs = map(f -> map(word -> positionIndex[lowercase(word)], f), fs)
    ys = map(context -> labelIndex[context.transition], contexts)
    # return a collection of tuples for this sentence, use Flux.batch to convert ws to a matrix of size 3 x (maxSequenceLength+1).
    # and xs to a matrix of size 4 x numberOfContexts
    # convert each output batch to an one-hot matrix of size (numLabels x numberOfContexts)
    ((Flux.batch(ws), Flux.batch(xs)), Flux.onehotbatch(ys, 1:length(labelIndex)))
end

"""
    batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)

    Create batches of data for training or evaluating. Each batch contains a pair (`Xb`, `Yb`) where 
    `Xb` is an array of `batchSize` samples. `Yb` is an one-hot matrix of size (`numLabels` x `batchSize`).
"""
function batch(sentences::Array{Sentence}, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int})
    # vectorizes all sentences 
    samples = map(sentence -> vectorize(sentence, wordIndex, shapeIndex, posIndex, labelIndex), sentences)
    X = map(sample -> sample[1], samples)
    Y = map(sample -> sample[2], samples)
    # build batches of data for training
    Xs = collect(Iterators.partition(X, options[:batchSize]))
    Ys = collect(Iterators.partition(Y, options[:batchSize]))
    (Xs, Ys)
end

"""
    eval(options, sentences)

    Evaluate the accuracy of the transition classifier.
"""
function eval(options, sentences::Array{Sentence})
    mlp, wordIndex, shapeIndex, posIndex, labelIndex = load(options)
    Xs, Ys = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)
    accuracy = evaluate(mlp, Xs, Ys)
    @info "accuracy = $(accuracy)"
    return accuracy
end

"""
    evaluate(mlp, Xs, Ys)

    Evaluate the accuracy of a model on a dataset.
"""
function evaluate(mlp, Xs, Ys)
    Ŷb = Iterators.flatten(map(X -> Flux.onecold.(mlp.(X)), Xs)) # numSentences arrays
    Yb = Iterators.flatten(map(Y -> [Flux.onecold(Y[i]) for i=1:length(Y)], Ys)) # numSentences arrays
    pairs = collect(zip(Ŷb, Yb))
    matches = map(p -> sum(p[1] .== p[2]), pairs)
    numSamples = sum(map(y -> length(y), Yb))
    accuracy = sum(matches)/numSamples
    return accuracy
end

"""
    train(options)

    Train a classifier model.
"""
function train(options)
    sentences = readCorpus(options[:trainCorpus], options[:maxSequenceLength])
    @info "#(sentencesTrain) = $(length(sentences))"
    contexts = collect(Iterators.flatten(map(sentence -> decode(sentence), sentences)))
    @info "#(contextsTrain) = $(length(contexts))"
    vocabularies = vocab(contexts)

    prepend!(vocabularies.words, [options[:unknown]])

    labelIndex = Dict{String, Int}(label => i for (i, label) in enumerate(vocabularies.labels))
    wordIndex = Dict{String, Int}(word => i for (i, word) in enumerate(vocabularies.words))
    shapeIndex = Dict{String, Int}(shape => i for (i, shape) in enumerate(vocabularies.shapes))
    posIndex = Dict{String, Int}(tag => i for (i, tag) in enumerate(vocabularies.partsOfSpeech))

    vocabSize = min(options[:vocabSize], length(wordIndex))
    @info "vocabSize = $(vocabSize)"

    # build training dataset
    Xs, Ys = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)
    dataset = collect(zip(Xs, Ys))
    @info "numBatches  = $(length(dataset))"
    # @info size(Xs[1][1][1]), size(Xs[1][1][2])
    # @info size(Ys[1][1])

    mlp = Chain(
        Join(
            EmbeddingWSP(vocabSize, options[:wordSize], length(shapeIndex), options[:shapeSize], length(posIndex), options[:posSize]),
            GRU(options[:wordSize] + options[:shapeSize] + options[:posSize], options[:embeddingSize])
            # GRU(options[:embeddingSize], options[:embeddingSize])
        ),
        Dense(options[:featuresPerContext] * options[:embeddingSize], options[:hiddenSize], tanh),
        Dense(options[:hiddenSize], length(labelIndex))
    )
    # save an index to an external file
    function saveIndex(index, path)
        file = open(path, "w")
        for f in keys(index)
            write(file, string(f, " ", index[f]), "\n")
        end
        close(file)
    end
    saveIndex(wordIndex, options[:wordPath])
    saveIndex(shapeIndex, options[:shapePath])
    saveIndex(posIndex, options[:posPath])
    saveIndex(labelIndex, options[:labelPath])

    # bring the dataset and the model to GPU if any
    if options[:gpu]
        @info "Bring data to GPU..."
        dataset = map(p -> p |> gpu, dataset)
        mlp = mlp |> gpu
    end
    # @info typeof(dataset[1][1]), size(dataset[1][1])
    # @info typeof(dataset[1][2]), size(dataset[1][2])

    @info "Total weight of initial word embeddings = $(sum(mlp[1].fs[1].word.W))"

    # build development dataset
    sentencesDev = readCorpus(options[:devCorpus], options[:maxSequenceLength])
    @info "#(sentencesDev) = $(length(sentencesDev))"
    contextsDev = collect(Iterators.flatten(map(sentence -> decode(sentence), sentencesDev)))
    @info "#(contextsDev) = $(length(contextsDev))"

    Xs, Ys = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)
    dataset = collect(zip(Xs, Ys))
    @info "numBatches  = $(length(dataset))"

    XsDev, YsDev = batch(sentencesDev, wordIndex, shapeIndex, posIndex, labelIndex)
    datasetDev = collect(zip(XsDev, YsDev))
    @info "numBatchesDev  = $(length(datasetDev))"

    # define a loss function, an optimizer and train the model
    function loss(X, Y)
        value = sum(Flux.logitcrossentropy(mlp(X[i]), Y[i]) for i=1:length(Y))
        Flux.reset!(mlp)
        return value
    end
    optimizer = ADAM()
    file = open(options[:logPath], "w")
    write(file, "dev. loss,trainingAcc,devAcc\n")
    evalcb = function()
        devLoss = sum(loss(datasetDev[i]...) for i=1:length(datasetDev))
        trainingAccuracy = evaluate(mlp, Xs, Ys)
        devAccuracy = evaluate(mlp, XsDev, YsDev)
        @info string("\tdevLoss = $(devLoss), training accuracy = $(trainingAccuracy), development accuracy = $(devAccuracy)")
        write(file, string(devLoss, ',', trainingAccuracy, ',', devAccuracy, "\n"))
    end
    # train the model until the development loss increases
    t = 1
    k = 0
    bestDevAccuracy = 0
    @time while (t <= options[:numEpochs])
        @info "Epoch $t, k = $k"
        Flux.train!(loss, params(mlp), dataset, optimizer, cb = Flux.throttle(evalcb, 60))
        devAccuracy = evaluate(mlp, XsDev, YsDev)
        if bestDevAccuracy < devAccuracy
            bestDevAccuracy = devAccuracy
            k = 0
        else
            k = k + 1
            if (k == 3)
                @info "Stop training because current accuracy is smaller than the best accuracy: $(devAccuracy) < $(bestDevAccuracy)."
                break
            end
        end
        @info "bestDevAccuracy = $bestDevAccuracy"
        t = t + 1
    end
    close(file)
    @info "Total weight of final word embeddings = $(sum(mlp[1].fs[1].word.W))"

    # evaluate the model on the training set
    # @info "Evaluating the model..."
    # accuracy = evaluate(mlp, Xs, Ys)
    # @info "Training accuracy = $accuracy"
    # accuracyDev = evaluate(mlp, XsDev, YsDev)
    # @info "Development accuracy = $(accuracyDev)"
    
    # save the model to a BSON file
    if (options[:gpu])
        mlp = mlp |> cpu
    end
    @save options[:modelPath] mlp
    mlp
end

"""
    dict(path)

    Load a dictionary (i.e., vocab or label) from a text file.
"""
function dict(path::String)::Dict{String,Int}
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    dict = Dict{String,Int}()
    for line in lines
        j = findlast(' ', line)
        if (j !== nothing)
            v = nextind(line, j)
            dict[strip(line[1:v-1])] = parse(Int, line[v:end])
        end
    end
    dict
end

"""
    load(options)

    Load a pre-trained classifier and return a tuple of (mlp, wordIndex, shapeIndex, posIndex, labelIndex).
"""
function load(options::Dict{Symbol,Any})::Tuple{Chain,Dict{String,Int},Dict{String,Int},Dict{String,Int},Dict{String,Int}}
    @load options[:modelPath] mlp
    wordIndex = dict(options[:wordPath])
    shapeIndex = dict(options[:shapePath])
    posIndex = dict(options[:posPath])
    labelIndex = dict(options[:labelPath])
    (mlp, wordIndex, shapeIndex, posIndex, labelIndex)
end
