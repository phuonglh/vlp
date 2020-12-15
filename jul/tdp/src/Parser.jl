include("Classifier.jl")

"""
    run(options)

    Parse an array of sentences. The pre-trained transition classifier, its associated feature index and label index 
    will be loaded from saved files in advance.
"""
function run(options::Dict{Symbol,Any}, sentences::Array{Sentence})
    mlp, featureIndex, labelIndex = load(options)
    # create (index => label) map
    labels = Dict{Int,String}(labelIndex[label] => label for label in keys(labelIndex))
    for sentence in sentences
        @info sentence
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
        while !isempty(β)
            features = featurize(config, tokenMap)
            x = unique(map(f -> get(featureIndex, f, 1), features))
            # pad x if necessary
            append!(x, ones(options[:featuresPerContext] - length(x)))
            y = mlp(x)
            transition = labels[Flux.onecold(y)[1]]
            config = next(config, transition)
        end
        # updated the predicted head (:h) and predicted dependency labels (:l) for all tokens of the current sentence
        for arc in config.arcs
            token = tokenMap[arc.dependent]
            token.annotation[:h] = arc.head
            token.annotation[:l] = arc.label
        end
    end
end

"""
    evaluate(options)

    Evaluate the accuracy of the parser: compute the UAS and LAS scores.
"""
function evaluate(options::Dict{Symbol,Any})::Tuple{Float64,Float64}
    sentences = readCorpus(options[:corpusPath])
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


@time uas, las = evaluate(options)
println("UAS = $uas")
println("LAS = $las")