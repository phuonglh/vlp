### (C) phuonglh
### Transition-based dependency parsing.
### December 10, 2020.


using Flux
using Flux: @epochs
using BSON: @save, @load
using CUDA

include("Oracle.jl")
include("Embedding.jl")

function model(options::Dict{Symbol,Any}, numLabels::Int)
    Chain(
        Embedding(options[:numFeatures], options[:embeddingSize]),
        Dense(options[:embeddingSize], options[:hiddenSize], σ),
        Dense(options[:hiddenSize], numLabels)
    )
end 

"""
    vocab(contexts, minFreq, makeLowercase)

    Builds two vocabularies of features and transitions. The feature vocabulary is sorted by frequency.
    Only features whose count is greater than `minFreq` are kept.
"""    
function vocab(contexts::Array{Context}, minFreq::Int = 2, makeLowercase::Bool = true)::Tuple{Array{String},Array{String}}
    features = Iterators.flatten(map(context -> context.features, contexts))
    transitions = map(context -> context.transition, contexts)
    frequency = Dict{String, Int}()
    for feature in features
        token = if makeLowercase lowercase(strip(feature)) else string(strip(feature)) end
        haskey(frequency, token) ? frequency[token] += 1 : frequency[token] = 1
    end
    # filter out infrequent tokens
    filter!(p -> p.second >= minFreq, frequency)
    # return a vocabulary of features and a vocabulary of transitions
    (collect(keys(frequency)), unique(transitions))
end

"""
    batch(contexts, featureIndex, labelIndex)

    Create batches of data for training or evaluating. Each batch contains a pair (Xb, Yb) where 
    Xb is a matrix of size (featuresPerContext x batchSize). Each column of Xb is a vector representing a bag of features extracted 
    from a context. If that number of features is less than `featuresPerContext`, this vector is padded with the [UNK] feature,
    which is `1`. Yb is an one-hot matrix of size (numLabels x batchSize).
"""
function batch(contexts::Array{Context}, featureIndex::Dict{String,Int}, labelIndex::Dict{String,Int})
    X, y = Array{Array{Int,1},1}(), Array{Int,1}()
    for context in contexts
        x = unique(map(f -> get(featureIndex, f, 1), context.features))
        # pad x if necessary
        append!(x, ones(options[:featuresPerContext] - length(x)))
        push!(X, x)
        push!(y, labelIndex[context.transition])
    end
    # build batches of data for training
    Xb = Iterators.partition(X, options[:batchSize])
    Yb = Iterators.partition(y, options[:batchSize])
    # stack each input batch to a 2-d matrix
    Xs = map(b -> Int.(Flux.batch(b)), Xb)
    # convert each output batch to an one-hot matrix
    Ys = map(b -> Flux.onehotbatch(b, 1:length(labelIndex)), Yb)
    (Xs, Ys)
end

"""
    train(options)

    Train a neural network transition classifier.
"""
function train(options::Dict{Symbol,Any})
    # load training data
    sentences = readCorpus(options[:trainCorpus], options[:maxSequenceLength])
    contexts = collect(Iterators.flatten(map(sentence -> decode(sentence), sentences)))
    @info "Number of sentencesTrain = $(length(sentences))"
    @info "Number of contextsTrain  = $(length(contexts))"
    # load development data
    sentencesDev = readCorpus(options[:devCorpus], options[:maxSequenceLength])
    contextsDev = collect(Iterators.flatten(map(sentence -> decode(sentence), sentencesDev)))
    @info "Number of sentencesDev = $(length(sentencesDev))"
    @info "Number of contextsDev  = $(length(contextsDev))"

    # build vocabulary and label list, truncate the vocabulary to the maximum number of features if necessary.
    vocabulary, labels = vocab(contexts, options[:minFreq], options[:lowercase])
    # add [UNK] symbol at index 1
    prepend!(vocabulary, ["[UNK]"])
    if length(vocabulary) < options[:numFeatures]
        options[:numFeatures] = length(vocabulary)
    else
        resize!(vocabulary, options[:numFeatures])
    end
    # build a feature index to map each feature to an id
    featureIndex = Dict{String, Int}(feature => i for (i, feature) in enumerate(vocabulary))
    # build a label index to map each transition to an id
    labelIndex = Dict{String, Int}(label => i for (i, label) in enumerate(labels))
    # save the vocabulary and label to external files
    file = open(options[:vocabPath], "w")
    for f in vocabulary
        write(file, string(f, " ", featureIndex[f]), "\n")
    end
    close(file)
    file = open(options[:labelPath], "w")
    for f in labels
        write(file, string(f, " ", labelIndex[f]), "\n")
    end
    close(file)
    # build training dataset
    Xs, Ys = batch(contexts, featureIndex, labelIndex)
    dataset = collect(zip(Xs, Ys))
    @info "numFeatures =  $(options[:numFeatures])"
    @info "numLabels = $(length(labels))"
    @info "numBatches (training) = $(length(dataset))"

    XsDev, YsDev = batch(contextsDev, featureIndex, labelIndex)
    datasetDev = collect(zip(XsDev, YsDev))
    @info "numBatches (development) = $(length(datasetDev))"

    # create a model 
    mlp = model(options, length(labels))

    # bring the dataset and the model to GPU if any
    if options[:gpu]
        dataset = map(p -> p |> gpu, dataset)
        mlp = mlp |> gpu
    end
    @info typeof(dataset[1][1]), size(dataset[1][1])
    @info typeof(dataset[1][2]), size(dataset[1][2])

    # define a loss function, an optimizer and train the model
    loss(x, y) = Flux.logitcrossentropy(mlp(x), y)
    optimizer = ADAM()
    file = open(options[:logPath], "w")
    write(file, "dev. loss,trainingAcc,devAcc\n")
    # evaluate the model on a dataset
    function accuracy(Xs, Ys, numContexts)
        Ŷb = Flux.onecold.(mlp.(Xs))
        Yb = Flux.onecold.(Ys)
        pairs = collect(zip(Ŷb, Yb))
        matches = map(p -> sum(p[1] .== p[2]), pairs)
        sum(matches)/numContexts
    end
    evalcb = function()
        devLoss = sum(loss(dataset[i]...) for i=1:length(datasetDev))
        trainAccuracy = accuracy(Xs, Ys, length(contexts))
        devAccuracy = accuracy(XsDev, YsDev, length(contextsDev))
        @info "\tdevLoss = $devLoss, trainAccuracy=$trainAccuracy, devAccuracy=$devAccuracy"
        write(file, string(devLoss, ',', trainAccuracy, ',', devAccuracy, "\n"))
    end
    # train the model until the development accuracy decreases 2 consecutive times
    t = 1
    k = 0
    bestDevAccuracy = 0
    @time while (t <= options[:numEpochs]) 
        @info "Epoch $t, k = $k"
        Flux.train!(loss, params(mlp), dataset, optimizer, cb = Flux.throttle(evalcb, 30))
        devAccuracy = accuracy(XsDev, YsDev, length(contextsDev))
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
    @info "Evaluating the model..."
    trainingAcc = accuracy(Xs, Ys, length(contexts))
    @info "Training accuracy = $trainingAcc"
    devAcc = accuracy(XsDev, YsDev, length(contextsDev))
    @info "Development accuracy = $devAcc"
    # save the model to a BSON file
    if options[:gpu]
        mlp = mlp |> cpu
    end
    @save options[:modelPath] mlp
    mlp
end


"""
    dict(path)

    Load a dictionary (vocab or label) from a text file.
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

    Load a pre-trained classifier and return a triple of (mlp, featureIndex, labelIndex).    
"""
function load(options::Dict{Symbol,Any})::Tuple{Chain,Dict{String,Int},Dict{String,Int}}
    @load options[:modelPath] mlp
    featureIndex = dict(options[:vocabPath])
    labelIndex = dict(options[:labelPath])
    (mlp, featureIndex, labelIndex)
end

"""
    eval(options, sentences)

    Evaluate the accuracy of the transition classifier.
"""
function eval(options::Dict{Symbol,Any}, sentences::Array{Sentence})
    contexts = collect(Iterators.flatten(map(sentence -> decode(sentence), sentences)))   
    @info "Number of sentences = $(length(sentences))"
    @info "Number of contexts  = $(length(contexts))"

    mlp, featureIndex, labelIndex = load(options)
    Xs, Ys = batch(contexts, featureIndex, labelIndex)
    dataset = collect(zip(Xs, Ys))
    @info "numFeatures = ", options[:numFeatures]
    @info "numBatches  = ", length(dataset)
    @info typeof(dataset[1][1]), size(dataset[1][1])
    @info typeof(dataset[1][2]), size(dataset[1][2])
    Ŷb = Flux.onecold.(mlp.(Xs))
    Yb = Flux.onecold.(Ys)
    pairs = collect(zip(Ŷb, Yb))
    matches = map(p -> sum(p[1] .== p[2]), pairs)
    numMatches = reduce((a, b) -> a + b, matches)
    @info numMatches
    accuracy = numMatches/length(contexts)
    @info "accuracy = $accuracy"
    mlp
end

# mlp = if options[:mode] == :train
#     train(options)
# elseif options[:mode] == :eval
#     sentencesTest = readCorpus(options[:testCorpus], options[:maxSequenceLength])
#     eval(options, sentencesTest)
# end
