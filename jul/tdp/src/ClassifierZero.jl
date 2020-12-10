
using Flux
using Flux: @epochs
using BSON: @save, @load


include("Oracle.jl")
include("Embedding.jl")

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
    # sort the word by frequency in decreasing order
    sort!(collect(frequency), by = p -> p.second, rev = true)
    # return a sorted vocabulary of features and a vocabulary of transitions
    (collect(keys(frequency)), unique(transitions))
end



options = Dict{Symbol,Any}(
    :minFreq => 2,
    :lowercase => true,
    :numFeatures => 2^16, 
    :embeddingSize => 100,
    :hiddenSize => 128,
    :batchSize => 32,
    :numEpochs => 50,
    :corpusPath => string(pwd(), "/dat/dep/eng/en-ud-dev.conllu"),
    :modelPath => string(pwd(), "/jul/tdp/dat/mlp.bson")
)

function model(options::Dict{Symbol,Any}, numLabels::Int)
    Chain(
        Dense(options[:embeddingSize], options[:hiddenSize], σ),
        Dense(options[:hiddenSize], numLabels)
    )
end 


"""
    train(options)

    Train a neural network transition classifier.
"""
function train(options::Dict{Symbol,Any})
    sentences = readCorpus(options[:corpusPath])
    contexts = collect(Iterators.flatten(map(sentence -> decode(sentence), sentences)))   
    @info "Number of sentences = $(length(sentences))"
    @info "Number of contexts  = $(length(contexts))"
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
    X, y = Array{Array{Int,1},1}(), Array{Int,1}()
    for context in contexts
        x = unique(map(f -> get(featureIndex, f, 1), context.features))
        push!(X, vec(x))
        push!(y, labelIndex[context.transition])
    end
    # feature embedding matrix
    embedding = Embedding(rand(Float32, options[:embeddingSize], options[:numFeatures]))
    # build batches of data for training
    Xb = Iterators.partition(X, options[:batchSize])
    # convert each input batch to a 2-d matrix of size batchSize x embeddingSize
    Xs = map(b -> Flux.batch([embedding(x) for x in b]), Xb)
    Yb = Iterators.partition(y, options[:batchSize])
    # convert each output batch to an one-hot matrix
    Ys = map(b -> Flux.onehotbatch(b, 1:length(labels)), Yb)
    dataset = collect(zip(Xs, Ys))
    @info "numFeatures = ", options[:numFeatures]
    @info "numBatches  = ", length(dataset)
    @info typeof(dataset[1][1]), size(dataset[1][1])
    @info typeof(dataset[1][2]), size(dataset[1][2])
    # define a model and a loss function
    mlp = model(options, length(labels))
    loss(x, y) = Flux.logitcrossentropy(mlp(x), y)
    optimizer = ADAM()
    evalcb = Flux.throttle(30) do
         @show(loss(dataset[1]...)) 
    end
    # train the model
    @epochs options[:numEpochs] Flux.train!(loss, params(mlp), dataset, optimizer, cb = evalcb)
    # evaluate the model on the training set
    Ŷb = Flux.onecold.(mlp.(Xs))
    pairs = collect(zip(Ŷb, Yb))
    matches = map(p -> sum(p[1] .== p[2]), pairs)
    accuracy = reduce((a, b) -> a + b, matches)/length(y)
    @info "Training accuracy = $accuracy"
    # save the model to a BSON file
    @save options[:modelPath] mlp
    (embedding, mlp)
end

function predict(options::Dict{Symbol,Any})
end

(embedding, mlp) = train(options)