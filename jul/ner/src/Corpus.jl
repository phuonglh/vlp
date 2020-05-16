#=
    A corpus contains a list of documents or sentences.
    phuonglh@gmail.com
=#

include("WordShape.jl")

struct Token
    text::String
    properties::Dict{Char, String}
end

struct Sentence
    tokens::Array{Token}
end

"Creates a token from a line in the CoNLL format. "
function token(line::String)::Token
    parts = string.(split(line, r"[\s]+"))
    properties = Dict{Char, String}()
    properties['p'] = parts[2]
    properties['c'] = parts[3]
    properties['e'] = parts[4]
    properties['s'] = shape(parts[1])
    Token(parts[1], properties)
end

"Reads a PoS tagged corpus in the VLSP format and returns an array of sentences."
function readVLSP(path::String)::Array{Sentence}
    sentences = Sentence[]
    lines = readlines(path)
    for line in lines
        wts = string.(split(strip(line), r"[\s]+"))
        tokens = Token[]
        for wt in wts
            properties = Dict{Char, String}()
            parts = string.(split(wt, r"/"))
            if (length(parts) == 2)
                properties['p'] = parts[2]
                properties['s'] = shape(parts[1])
                token = Token(parts[1], properties)
                push!(tokens, token)
            else
                j = findlast("/", wt)[1]
                if (j == length(wt)) # the case ///
                    properties['p'] = "/"
                    properties['s'] = shape("/")
                    push!(tokens, Token("/", properties))
                else
                    w = wt[1:j-1]
                    properties['p'] = wt[j+1:end]
                    properties['s'] = shape(w)
                    push!(tokens, Token(w, properties))
                end
            end            
        end
        push!(sentences, Sentence(tokens))
    end
    sentences
end

"Reads a NER corpus in the CoNLL format and returns an array of sentences. "
function readCoNLL(path::String)::Array{Sentence}
    sentences = Sentence[]
    lines = readlines(path)
    n = length(lines)
    indexedLines = collect(zip(1:n, map(line -> strip(line), lines)))
    emptyIndices = map(p -> p[1], filter(p -> isempty(p[2]), indexedLines))
    j = 1
    for i in emptyIndices
        xs = lines[j:i-1]
        if (isempty(xs))
            println("Problematic line: ", i)
        end
        tokens = map(x -> token(x), xs)
        push!(sentences, Sentence(tokens))
        j = i+1
    end
    sentences
end

"Builds vocabulary from a list of sentences. A vocabulary is an array of tokens which are sorted by frequency."
function vocab(sentences::Array{Sentence}, minFreq::Int = 2, makeLowercase::Bool = true)::Array{String}
    frequency = Dict{String, Int}()
    for sentence in sentences
        for token in sentence.tokens
            word = if makeLowercase lowercase(strip(token.text)) else string(strip(token.text)) end
            if (shape(word) == "number") word = "<number>" end
            if !isempty(word)
                haskey(frequency, word) ? frequency[word] += 1 : frequency[word] = 1
            end
        end
    end
    # filter out infrequent words
    filter!(p -> p.second >= minFreq, frequency)
    # sort the word by frequency in decreasing order
    sort!(collect(frequency), by = p -> p.second, rev = true)
    # return the vocabulary
    collect(keys(frequency))
end

"Builds an label array from a list of sentences."
function labels(sentences::Array{Sentence}, c::Char)::Array{String}
    xs = map(sentence -> map(token -> token.properties[c], sentence.tokens), sentences)
    unique(Iterators.flatten(xs))
end

"Computes length statistics of sentences in the corpus."
function lengthFreq(sentences::Array{Sentence})::Dict{Int, Int}
    stats = Dict{Int, Int}()
    for s in sentences
        x = length(s.tokens)
        haskey(stats, x) ? stats[x] += 1 : stats[x] = 1
    end
    stats
end