

include("WordShape.jl")
include("Mutate.jl")

"""
Builds a vocabulary of syllables from input sentences: word => frequency
"""
function vocab(sentences::Array{String}, minFreq::Int = 2, makeLowercase::Bool = true)::Dict{String,Int}
frequency = Dict{String,Int}()
for sentence in sentences
    tokens = split(sentence, delimiters)
    for token in tokens
        word = if makeLowercase lowercase(strip(token)) else string(strip(token)) end
        s = shape(word)
#            if (s == "number") word = "0" end
        if (s == "email") word = "phuonglh@gmail.com" end
        if (s == "url") word = "http://phuonglh.com" end
        if !isempty(word)
            haskey(frequency, word) ? frequency[word] += 1 : frequency[word] = 1
        end
    end
end
# filter out infrequent words
filter!(p -> p.second >= minFreq, frequency)
frequency
end
