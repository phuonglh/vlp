

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

