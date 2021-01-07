using FLoops

include("Classifier.jl")

"""
    run(options)

    Parse an array of sentences. A pre-trained transition classifier, its associated indices 
    will be loaded from saved files in advance. The input sentences are updated in-place by appending :h and :l annotations to 
    each token for head and dependency label predictions.
"""
function run(options::Dict{Symbol,Any}, sentences::Array{Sentence})
    mlp, wordIndex, shapeIndex, posIndex, labelIndex = load(options)
    # create (index => label) map
    labels = Dict{Int,String}(labelIndex[label] => label for label in keys(labelIndex))
    # parse the sentences in parallel to speed up the processing
    @floop ThreadedEx(basesize=length(sentences)÷options[:numCores]) for sentence in sentences
        σ = Stack{String}()
        β = Queue{String}()
        tokenMap = Dict{String,Token}(token.annotation[:id] => token for token in sentence.tokens)
        for id in map(token -> token.annotation[:id], sentence.tokens)
            enqueue!(β, id)
        end
        A = Array{Arc,1}()
        config = Config(σ, β, A)
        contexts = []
        config = next(config, "SH")
        transition = "SH"
        # create a word map to position
        words = map(token -> lowercase(token.word), sentence.tokens)
        append!(words, [options[:padding]])
        positionIndex = Dict{String,Int}(word => i for (i, word) in enumerate(words))
        # convert sentence to a 3 x maxSequenceLength integer matrix
        ws = vectorizeSentence(sentence, wordIndex, shapeIndex, posIndex)
        as = Flux.batch(ws)
        # the greedy parsing loop
        while !isempty(β) && !isempty(σ)
            u, v = tokenMap[first(σ)], tokenMap[first(β)]
            ws0 = lowercase(u.word)
            wq0 = lowercase(v.word)
            wq1 = if length(β) > 1
                id = collect(β)[2]
                v = tokenMap[id]
                lowercase(v.word)
            else
                "[pad]"
            end
            ws1 = if length(σ) > 1
                id = collect(σ)[2]
                v = tokenMap[id]
                lowercase(v.word)
            else
                "[pad]"
            end
            fs = [ws0; wq0; wq1; ws1] # this creates a vector of length 4
            bs = map(word -> positionIndex[lowercase(word)], fs)
            x = (as, bs) # input to our model is a pair of (matrix, vector)
            y = mlp(x)
            transition = labels[Flux.onecold(y)[1]]
            config = next(config, transition)
        end
        # updated the predicted head (:h) and predicted dependency labels (:l) for all tokens
        for arc in config.arcs
            token = tokenMap[arc.dependent]
            token.annotation[:h] = arc.head
            token.annotation[:l] = arc.label
        end
    end
end

"""
    evaluate(options, sentences)

    Evaluate the accuracy of the parser: compute the UAS and LAS scores.
"""
function evaluate(options::Dict{Symbol,Any}, sentences::Array{Sentence})::Tuple{Float64,Float64}
    run(options, sentences)
    uas, las = 0, 0
    numTokens = 0
    for sentence in sentences
        numTokens += length(sentence.tokens)
        for token in sentence.tokens
            if (:h ∈ keys(token.annotation))
                uas += (token.annotation[:head] == token.annotation[:h])
                if (:l ∈ keys(token.annotation))
                    las += (token.annotation[:head] == token.annotation[:h]) && (token.annotation[:label] == token.annotation[:l])
                end
            end
        end
    end
    (uas, las) ./ numTokens
end

"""
    evaluate(options)

    Evaluate the performance of the parser on all train/dev./test datasets.
"""
function evaluate(options)
    sentences = readCorpus(options[:trainCorpus], options[:maxSequenceLength])
    @time uas, las = evaluate(options, sentences)
    @info "Training scores: UAS = $uas, LAS = $las"

    sentences = readCorpus(options[:devCorpus], options[:maxSequenceLength])
    @time uas, las = evaluate(options, sentences)
    @info "Development scores: UAS = $uas, LAS = $las"

    sentences = readCorpus(options[:testCorpus], options[:maxSequenceLength])
    @time uas, las = evaluate(options, sentences)
    @info "Test scores: UAS = $uas, LAS = $las"
end
