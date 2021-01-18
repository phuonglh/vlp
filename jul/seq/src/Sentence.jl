include("Brick.jl")

struct Token
    word::String
    annotation::Dict{Symbol,String}
end

struct Sentence
    tokens::Array{Token}
end

"""
    readCorpus(path, maxSentenceLength=40)

    Read a CoNLLU file to build dependency graphs. Each graph is a sentence.
"""
function readCorpusUD(path::String, maxSentenceLength::Int=40)::Array{Sentence}
    lines = filter(line -> !startswith(line, "#"), readlines(path))
    append!(lines, [""])
    sentences = []
    tokens = []
    for line in lines
        parts = split(strip(line), r"\t+")
        if length(parts) == 1
            prepend!(tokens, [Token("ROOT", Dict(:id => "0", :lemma => "NA", :upos => "NA", :pos => "NA", :fs => "NA", :head => "NA", :label => "NA"))])
            # add sentence if it is not too long...
            if length(tokens) <= maxSentenceLength
              push!(sentences, Sentence(tokens))
            end
            empty!(tokens)
        else
            word = parts[2]
            fullLabel = parts[8]
            colonIndex = findfirst(':', fullLabel)
            label = if (colonIndex !== nothing) fullLabel[1:colonIndex-1] else fullLabel end
            annotation = Dict(:id => parts[1], :lemma => parts[3], :upos => parts[4], :pos => parts[5], :fs => parts[6], :head => parts[7], :label => label)
            push!(tokens, Token(word, annotation))
        end
    end
    sentences
end

"""
    readCorpus(path, maxSentenceLength=40)

    Read a CoNLL-2003 file to build named-entity tagged sentences.
"""
function readCorpusCoNLL(path::String, maxSentenceLength::Int=40)::Array{Sentence}
    function createToken(line::String)::Token
        parts = string.(split(line, r"[\s]+"))
        annotation = Dict(:p => parts[2], :c => parts[3], :e => parts[4], :s => shape(parts[1]))
        Token(parts[1], annotation)
    end

    sentences = []
    lines = readlines(path)
    n = length(lines)
    indexedLines = collect(zip(1:n, map(line -> strip(line), lines)))
    emptyIndices = map(p -> p[1], filter(p -> isempty(p[2]), indexedLines))
    j = 1
    for i in emptyIndices
        xs = lines[j:i-1]
        if (isempty(xs))
            @warn ("Problematic line: $i")
        end
        tokens = createToken.(xs)
        push!(sentences, Sentence(tokens))
        j = i+1
    end
    sentences
end
