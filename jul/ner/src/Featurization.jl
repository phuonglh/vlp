"""
    phuonglh
    November 8, 2019
"""

using Flux: onehot, onehotbatch

include("WordVectors.jl")
include("Embedding.jl")

wd = 50
makeLowercase = true

# build vocab and embedding matrix
wordList = vocab(sentences, minFreq, makeLowercase)
push!(wordList, "<number>")
push!(wordList, "UNK")
println("#(vocab) = ", length(wordList))

# build word vectors
@time wordVectors = load("/opt/data/emb/skip.vie.50d.txt")

# prepare the word embedding table
N = length(wordList)
W = rand(wd, N)
for i = 1:N
    word = wordList[i]
    if (haskey(wordVectors, word))
        W[:, i] = wordVectors[word]
    end
end

embed = Embedding(W)

# build a word index (word => index)
wordIndex = Dict{String, Int}(word => i for (i, word) in enumerate(wordList))

entities = labels(sentences, 'e')
println("Entity types = ", entities)

partsOfSpeech = labels(sentences, 'p')
chunkTypes = labels(sentences, 'c')
wordShapes = labels(sentences, 's')

"""
Creates a matrix representation of a sentence. Each sequence of N tokens 
is converted to a matrix of size DxN, each column vector coressponds to a token.
"""
function vectorize(sentence::Sentence, training::Bool = false)
    tokens = sentence.tokens
    # word indices for embeddings
    ws = Array{Int,1}()
    for i = 1:length(tokens)
        word = lowercase(tokens[i].text)
        if (haskey(wordIndex, word))
            push!(ws, wordIndex[word])
        elseif (shape(word) == "number")
            push!(ws, wordIndex["<number>"])
        else
            push!(ws, wordIndex["UNK"])
        end
    end
    # one-hot parts-of-speech vector
    pos = map(token -> token.properties['p'], tokens)
    ps = onehotbatch(pos, partsOfSpeech)
    # one-hot chunk type vector
    chs = map(token -> token.properties['c'], tokens)
    cs = onehotbatch(chs, chunkTypes)
    # one-hot shape vector
    shs = map(token -> token.properties['s'], tokens)
    ss = onehotbatch(shs, wordShapes)
    # convert xs to Float32 to speed up computation
    xs = Float32.(vcat(embed(ws), ps, cs, ss))
    if (training)
        yy = map(token -> token.properties['e'], tokens)
        ys = onehotbatch(yy, entities)
        (xs, Float32.(ys))
    else
        xs
    end    
end

