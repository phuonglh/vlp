# phuonglh@gmail.com
# Semi-character GRU for VSC

using Flux
using Flux: @epochs, onehotbatch, throttle, crossentropy, reset!, onecold
using BSON: @save, @load

include("Options.jl")

"""
    readData(options)

    Read training data from a text file, return an array of sentences and their corresponding arrays of mutations.
"""
function readData(options)
    lines = readlines(options[:inputPath])
    sentences = Array{Array{String,1},1}()
    mutations = Array{Array{Symbol,1},1}()
    i = 1
    while i < length(lines)
        y = map(a -> Symbol(a), split(lines[i], " "))
        push!(mutations, y)
        x = string.(split(lines[i+1], " "))
        push!(sentences, x)
        i = i + 2
    end
    return (sentences, mutations)
end
  

"""
  vectorize(tokens, alphabet, mutations)

  Transforms a sentence into an array of vectors based on an alphabet and 
  a label set. Return a pair of matrix (x, y) in the training mode or a single input matrix x in the test mode, where
  x and y are matrices of size `(3*|alphabet|) x length(x)`, where length(x) is the number of tokens of the sentence.
"""
function vectorize(tokens::Array{String}, alphabet::Array{Char}, mutations::Array{Symbol})
    # Compute bag-of-character vectors for middle characters of a token, that is token[2:end-1].
    function boc(token::String, alphabet::Array{Char})
        a = onehotbatch(collect(token[nextind(token, 1):prevind(token, lastindex(token))]), alphabet)
        sum(a, dims=2)
    end
    # truncate or padding input sequences to have the same maxSequenceLength
    n = options[:maxSequenceLength]
    (x, y) = if length(tokens) >= n
        (tokens[1:n], mutations[1:n])
    else
        for t=1:(n-length(tokens))
            push!(tokens, "abc")
            push!(mutations, :P)
        end
        (tokens, mutations)
    end
    # one-hot vectors of the first chars for each token 
    ucs = map(token -> first(token), x)
    us = onehotbatch(ucs, alphabet)
    # one-hot vectors of the last chars for each tokens
    vcs = map(token -> last(token), x)
    vs = onehotbatch(vcs, alphabet)
    # middle bag-of-character vectors
    cs = zeros(length(alphabet), length(x))
    for j = 1:length(x)
        cs[:,j] = boc(x[j], alphabet)
    end
    # combine all vectors into xs and convert xs to Float32 to speed up computation
    xs = Float32.(vcat(us, vs, cs))
    # prepare one-hot vectors for labels
    ys = onehotbatch(y, options[:labels])
    return (Float32.(xs), Float32.(ys))
end

"""
    batch(sentences, alphabet, mutations)

    Create batches of data for training.
"""
function batch(sentences::Array{Array{String,1}}, alphabet::Array{Char,1}, mutations::Array{Array{Symbol,1}})
    pairs = [vectorize(sentences[i], alphabet, mutations[i]) for i=1:length(sentences)]
    Xs = map(p -> p[1], pairs)
    Ys = map(p -> p[2], pairs)
    Xb = collect(Iterators.partition(Xs, options[:batchSize]))
    Yb = collect(Iterators.partition(Ys, options[:batchSize]))
    return (Xb, Yb)
end

"""
  predict(model, sentence, alphabet)

  Predict a mutated sentence.
"""
function predict(model, sentence::Array{String}, alphabet::Array{Char})::Array{Symbol}
    # reset the state of the model before applying on an input sample
    reset!(model) 
    # vectorize the input sentence using a dummy label sequence (not used)
    x = vectorize(sentence, alphabet, fill(:P, length(sentence)))
    y = onecold(model(x[1]))
    map(e -> options[:labels][e], y)
end

"""
  evaluate(model, xs, ys)

  Predict a list of sentences, collect prediction result and report prediction accuracy
  xs: mutated sentences; ys: correct mutation labels
"""
function evaluate(model, xs::Array{Array{String,1}}, alphabet::Array{Char}, ys::Array{Array{Symbol,1}})
    zs = map(s -> predict(model, s, alphabet), xs)
    oui = Dict{Symbol,Int}() # number of correct predictions for each label
    non = Dict{Symbol,Int}() # number of incorrect predictions for each label
    foreach(k -> oui[k] = 0, options[:labels])
    foreach(k -> non[k] = 0, options[:labels])
    for i = 1:length(ys)    
        y, z = ys[i], zs[i]
        zyDiff = (z .== y)
        for i = 1:length(y)
        k = y[i]
        if (zyDiff[i])
            oui[k] = oui[k] + 1
        else 
            non[k] = non[k] + 1
        end
        end
    end
    accuracy = Dict{Symbol,Float64}()
    for k in options[:labels]
        accuracy[k] = oui[k]/(oui[k] + non[k])
    end
    @info "\toui = $(oui)"
    @info "\tnon = $(non)"
    return accuracy
end

"""
    evaluate(model, xs, ys)

    Evaluates the accuracy of a mini-batch, return the total labels in the batch and the number of correct predicted labels.
    This function is used in traiing for performance update.
"""
function evaluate(model, xs::Array{Array{Float32,2}}, ys::Array{Array{Float32,2}})::Tuple{Int,Int}
    as = map(x -> onecold(model(x)), xs)
    bs = map(y -> onecold(y), ys)
    # find the real length of target sequence (without padding symbol :P)
    total = 0
    correct = 0
    padding = length(options[:labels])
    for i = 1:length(as)
        t = options[:maxSequenceLength]
        while t > 0 && bs[i][t] == padding
            t = t - 1
        end
        total = total + t
        correct = correct + sum(as[i][1:t] .== bs[i][1:t])
    end
    reset!(model)
    return (total, correct)
end

"""
    train(options)

    Trains a model and save to an external file; the alphabet is also saved.
"""
function train(options)
    sentences, mutations = readData(options)
    xs = replace!.(sentences, r"\s+" => "")
    alphabet = unique(join(join.(xs)))
    file = open(options[:alphabetPath], "w")
    write(file, join(alphabet))
    close(file)
    Xb, Yb = batch(sentences, alphabet, mutations)
    if options[:gpu] 
        @info "Bringing data to GPU..."
        Xb = map(t -> gpu.(t), Xb)
        Yb = map(t -> gpu.(t), Yb)
    end
    dataset = collect(zip(Xb, Yb))
    @info "#(batches) = $(length(dataset))"
    @info "typeof(X1) = $(typeof(Xb[1]))" # should be Array{Array{Float32,2},1}
    @info "typeof(Y1) = $(typeof(Yb[1]))" # should be Array{Array{Float32,2},1}
    # define a model
    model = Chain(
        GRU(3*length(alphabet), options[:hiddenSize]), 
        Dense(options[:hiddenSize], length(options[:labels])),
        softmax
    )
    @info model
    # compute the loss of the model on a batch
    function loss(Xs, Ys)
        value = sum(crossentropy(model(Xs[i]), Ys[i]) for i=1:length(Xs))
        reset!(model)
        return value
    end
    optimizer = ADAM()
    
    function evalcb() 
        xs, ys = collect(Xb[1]), collect(Yb[1])
        a = evaluate(model, xs, ys)
        @info "loss = $(loss(dataset[1]...)), accuracy = $a"
    end
    @epochs options[:numEpochs] Flux.train!(loss, params(model), dataset, optimizer, cb = throttle(evalcb, 60))
    if (options[:gpu])
        model = model |> cpu
    end
    @save options[:modelPath] model
    # evaluate the training accuracy of the model
    pairs = [evaluate(model, collect(Xb[i]), collect(Yb[i])) for i=1:length(Xb)]
    result = reduce(((a, b), (c, d)) -> (a + c, b + d), pairs)
    @info "training accuracy = $(result[2]/result[1]) [$(result[2])/$(result[1])]."
    return model
end

"""
    eval(options)

    Loads a trained model and evaluate the accuracy on training set and test set. 
"""
function eval(options)
    @load options[:modelPath] model
    sentences, mutations = readData(options)
    alphabet = collect(readline(options[:alphabetPath]))
    Xb, Yb = batch(sentences, alphabet, mutations)
    # evaluate the training accuracy of the model
    pairs = [evaluate(model, collect(Xb[i]), collect(Yb[i])) for i=1:length(Xb)]
    result = reduce(((a, b), (c, d)) -> (a + c, b + d), pairs)
    reset!(model)
    @info "training accuracy = $(result[2]/result[1]) [$(result[2])/$(result[1])]."
    evaluate(model, sentences, alphabet, mutations)
end

