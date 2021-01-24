# phuonglh@gmail.com
# Sentence encoder which encodes a sequence of tokens into a sequence of 
# dense vectors and perform sequence tagging. This programme performs part-of-speech 
# tagging on a Universal Dependencies treebank data. Here, we use (word, shape, universal PoS) to 
# infer language-specific part-of-speech.

using Flux
using Flux: @epochs
using BSON: @save, @load

using FLoops

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
        partsOfSpeech[token.annotation[:upos]] = 0
        labels[token.annotation[:pos]] = 0
    end
    # filter out infrequent words
    filter!(p -> p.second >= minFreq, wordFrequency)
    
    Vocabularies(collect(keys(wordFrequency)), collect(keys(shapes)), collect(keys(partsOfSpeech)), collect(keys(labels)))
end

"""
    batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex, options)

    Create batches of data for training or evaluating. Each batch contains a pair (Xb, Yb) where 
    Xb is an array of matrices of size (featuresPerToken x maxSequenceLength). Each column of Xb is a vector representing a token.
    If a sentence is shorter than maxSequenceLength, it is padded with vectors of ones. To speed up the computation, Xb and Yb 
    are stacked as 3-d matrices where the 3-rd dimention is the batch one.
"""
function batch(sentences::Array{Sentence}, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int}, options=optionsVUD)
    X, Y = Array{Array{Int,2},1}(), Array{Array{Int,2},1}()
    paddingX = [wordIndex[options[:paddingX]]; 1; 1]
    paddingY = Flux.onehot(labelIndex[options[:paddingY]], 1:length(labelIndex))
    for sentence in sentences
        xs = map(token -> [get(wordIndex, lowercase(token.word), 1), get(shapeIndex, shape(token.word), 1), get(posIndex, token.annotation[:upos], 1)], sentence.tokens)
        ys = map(token -> Flux.onehot(labelIndex[token.annotation[:pos]], 1:length(labelIndex), 1), sentence.tokens)
        # pad the columns of xs and ys to maxSequenceLength
        if length(xs) > options[:maxSequenceLength]
            xs = xs[1:options[:maxSequenceLength]]
            ys = ys[1:options[:maxSequenceLength]]
        end
        for t=length(xs)+1:options[:maxSequenceLength]
            push!(xs, paddingX) 
            push!(ys, paddingY)
        end
        push!(X, Flux.batch(xs))
        push!(Y, Flux.batch(ys))
    end
    # build batches of data for training
    Xb = Iterators.partition(X, options[:batchSize])
    Yb = Iterators.partition(Y, options[:batchSize])
    # stack each input batch as a 3-d matrix
    Xs = map(b -> Int.(Flux.batch(b)), Xb)
    # stack each output batch as a 3-d matrix
    Ys = map(b -> Int.(Flux.batch(b)), Yb)
    (Xs, Ys)
end

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
    train(options)

    Train an encoder.
"""
function train(options::Dict{Symbol,Any})
    sentences = readCorpusUD(options[:trainCorpus])
    sentencesValidation = readCorpusUD(options[:validCorpus])
    @info "Number of training sentences = $(length(sentences))"
    @info "Number of validation sentences = $(length(sentencesValidation))"
    vocabularies = vocab(sentences)
    
    prepend!(vocabularies.words, [options[:unknown]])
    append!(vocabularies.words, [options[:paddingX]])
    wordIndex = Dict{String,Int}(word => i for (i, word) in enumerate(vocabularies.words))
    shapeIndex = Dict{String,Int}(shape => i for (i, shape) in enumerate(vocabularies.shapes))
    posIndex = Dict{String,Int}(pos => i for (i, pos) in enumerate(vocabularies.partsOfSpeech))
    prepend!(vocabularies.labels, [options[:paddingY]])
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

    # save the vocabulary, shape, part-of-speech and label information to external files
    saveIndex(wordIndex, options[:wordPath])
    saveIndex(shapeIndex, options[:shapePath])
    saveIndex(posIndex, options[:posPath])
    saveIndex(labelIndex, options[:labelPath])

    # define a model for sentence encoding
    encoder = Chain(
        EmbeddingWSP(min(length(wordIndex), options[:vocabSize]), options[:wordSize], length(shapeIndex), options[:shapeSize], length(posIndex), options[:posSize]),
        GRU(options[:wordSize] + options[:shapeSize] + options[:posSize], options[:hiddenSize]),
        Dense(options[:hiddenSize], length(labelIndex))
    )

    @info "Total weight of initial word embeddings = $(sum(encoder[1].word.W))"

    """
        loss(X, Y)

        Compute the loss on one batch of data where X and Y are 3-d matrices of size (K x maxSequenceLength x batchSize).
        We use the log cross-entropy loss to measure the average distance between prediction and true sequence pairs.
    """
    function loss(X, Y)
        b = size(X,3)
        predictions = [encoder(X[:,:,i]) for i=1:b]
        truths = [Y[:,:,i] for i=1:b]
        value = sum(Flux.logitcrossentropy(predictions[i], truths[i]) for i=1:b)
        Flux.reset!(encoder)
        return value
    end

    Us, Vs = batch(sentencesValidation, wordIndex, shapeIndex, posIndex, labelIndex)
    datasetValidation = collect(zip(Us, Vs))

    optimizer = ADAM()
    file = open(options[:logPath], "w")
    write(file, "loss,trainingAccuracy,validationAccuracy\n")
    evalcb = Flux.throttle(30) do
        ℓ = loss(dataset[1]...)
        trainingAccuracy = evaluate(encoder, Xs, Ys)
        validationAccuracy = evaluate(encoder, Us, Vs)
        @info string("loss = ", ℓ, ", training accuracy = ", trainingAccuracy, ", validation accuracy = ", validationAccuracy)
        write(file, string(ℓ, ',', trainingAccuracy, ',', validationAccuracy, "\n"))
    end
    # train the model
    @time @epochs options[:numEpochs] Flux.train!(loss, params(encoder), dataset, optimizer, cb = evalcb)
    close(file)
    # save the model to a BSON file
    @save options[:modelPath] encoder

    @info "Total weight of final word embeddings = $(sum(encoder[1].word.W))"
    @info "Evaluating the model on the training set..."
    accuracy = evaluate(encoder, Xs, Ys)
    @info "Training accuracy = $accuracy"
    accuracyValidation = evaluate(encoder, Us, Vs)
    @info "Validation accuracy = $accuracyValidation"
    encoder
end

"""
    evaluate(encoder, Xs, Ys, paddingY)

    Evaluate the accuracy of the encoder on a dataset. `Xs` is a list of 3-d input matrices and `Ys` is a list of 
    3-d ground-truth output matrices. The third dimension is the batch one.
"""
function evaluate(encoder, Xs, Ys, paddingY::Int=1)
    numBatches = length(Xs)
    # normally, size(X,3) is the batch size except the last batch
    @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
        b = size(Xs[i],3)
        Flux.reset!(encoder)
        Ŷb = Flux.onecold.(encoder(Xs[i][:,:,t]) for t=1:b)
        Yb = Flux.onecold.(Ys[i][:,:,t] for t=1:b)
        # number of tokens and number of matches in this batch
        tokens, matches = 0, 0
        for t=1:b
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
    @info "Total matched tokens = $(numMatches)/$(numTokens)"
    return numMatches/numTokens
end
