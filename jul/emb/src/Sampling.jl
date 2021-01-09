# phuonglh@gmail.com
# A tool to collect all dependency samples in the format of 
# a list (head, tail/dependent, label), sorted by heads.
# These triples are written to a file, each line begins with a head, a \t character, 
# and a set of all tail/label relation separated by the space character.

include("../../tdp/src/Sentence.jl")
include("../../tdp/src/Options.jl")

struct Triplet
    head::String
    tail::String
    label::String
end

"""
    extractTriplets(options)

    Extracts all triplets from the training set, return a list of words (vocabulary) and a set of triplets.
"""
function extractTriplets(options)::Tuple{Array{String},Set{Triplet}}
    sentences = readCorpus(options[:trainCorpus], options[:maxSequenceLength])
    words = Set{String}()
    triplets = Set{Triplet}()
    for sentence in sentences
        id2Words = Dict{String,String}()
        for token in sentence.tokens
            id2Words[token.annotation[:id]] = lowercase(replace(token.word, " " => "_"))        
        end
        for token in sentence.tokens
            tail = lowercase(replace(token.word, " " => "_"))
            head = get(id2Words, token.annotation[:head], "NA")
            label = token.annotation[:label]
            push!(triplets, Triplet(head, tail, label))
            push!(words, head)
            push!(words, tail)
        end
    end
    (sort!(collect(words)), triplets)
end

"""
    headToTails(options)

    Build an external file of format `head -> [tail/label...]` to investigate 
    the outgoing relationships from a given word.
"""
function headToTails(options)
    sentences = readCorpus(options[:trainCorpus], options[:maxSequenceLength])
    s = Dict{String,Set{String}}()
    for sentence in sentences
        id2Words = Dict{String,String}()
        for token in sentence.tokens
            id2Words[token.annotation[:id]] = lowercase(replace(token.word, " " => "_"))
        end
        for token in sentence.tokens
            tail = lowercase(replace(token.word, " " => "_"))
            head = get(id2Words, token.annotation[:head], "NA")
            label = token.annotation[:label]
            value = get(s, head, Set{String}())
            push!(value, string(tail, "/", label))
            s[head] = value
        end
    end
    words = sort!(collect(keys(s)))
    lines = map(word -> string(word, '\t', join(s[word], " ")), words)
    file = open(options[:statPath], "w")
    foreach(line -> write(file, string(line, "\n")), lines)
    close(file)
    @info "$(length(lines)) lines are written to $(options[:statPath])."
end
