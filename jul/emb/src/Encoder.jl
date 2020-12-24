# phuonglh@gmail.com
# Sentence encoder which encodes a sequence of tokens into a sequence of 
# dense vectors. 

using Flux

include("Embedding.jl")
include("Options.jl")
include("../../tdp/src/Sentence.jl")
include("../../tok/src/Brick.jl")


struct Vocabularies
    words::Array{String}
    shapes::Array{String}
    partsOfSpeech::Array{String}
    labels::Array{String}
end

"""
    vocab(contexts, minFreq)

    Builds vocabularies of words, shapes, parts-of-speech, and labels. The word vocabulary is sorted by frequency.
    Only words whose count is greater than `minFreq` are kept.
"""    
function vocab(sentences::Array{Sentence}, minFreq::Int = 2)::Vocabularies
    tokens = Iterators.flatten(map(sentence -> sentence.tokens, sentences))
    wordFrequency = Dict{String, Int}()
    shapes = Dict{String,Int}()
    partsOfSpeech = Dict{String,Int}()
    labels = Dict{String,Int}()
    for token in tokens
        word = lowercase(strip(token.word))
        haskey(wordFrequency, word) ? wordFrequency[word] += 1 : wordFrequency[word] = 1
        shapes[shape(token.word)] = 0
        partsOfSpeech[token.annotation[:upos]] = 0
        labels[token.annotation[:head]] = 0
    end
    # filter out infrequent words
    filter!(p -> p.second >= minFreq, wordFrequency)
    
    Vocabularies(collect(keys(wordFrequency)), collect(keys(shapes)), collect(keys(partsOfSpeech)), collect(keys(labels)))
end

"""
    batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)

    Create batches of data for training or evaluating. Each batch contains a pair (Xb, Yb) where 
    Xb is an array of matrices of size (featuresPerToken x maxSequenceLength). Each column of Xb is a vector representing a token.
    If a sentence is shorter than maxSequenceLength, it is padded with vectors of ones.
"""
function batch(sentences::Array{Sentence}, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int})
    X, Y = Array{Array{Int,2},1}(), Array{Array{Int,2},1}()
    for sentence in sentences
        xs = map(token -> [get(wordIndex, lowercase(token.word), 1), shapeIndex[shape(token.word)], posIndex[token.annotation[:upos]]], sentence.tokens)
        ys = map(token -> Flux.onehot(labelIndex[token.annotation[:head]], 1:length(labelIndex), 1), sentence.tokens)
        # pad the columns of xs and ys to maxSequenceLength
        if length(xs) > options[:maxSequenceLength]
            xs = xs[1:options[:maxSequenceLength]]
            ys = ys[1:options[:maxSequenceLength]]
        end
        for t=length(xs)+1:options[:maxSequenceLength]
            push!(xs, ones(3))
            push!(ys, Flux.onehot(1, 1:length(labelIndex)))
        end
        push!(X, Flux.batch(xs))
        push!(Y, Flux.batch(ys))
    end
    # build batches of data for training
    Xb = Iterators.partition(X, options[:batchSize])
    Yb = Iterators.partition(Y, options[:batchSize])
    # stack each input batch to a 3-d matrix
    Xs = map(b -> Int.(Flux.batch(b)), Xb)
    # stack each output batch to a 3-d matrix
    Ys = map(b -> Int.(Flux.batch(b)), Yb)
    (Xs, Ys)
end

sentences = readCorpus(options[:trainCorpus])
vocabularies = vocab(sentences)

# encoder = Chain(
#     EmbeddingWSP(min(length(wordVocab), options[:vocabSize]), options[:wordSize], length(shapeVocab), options[:shapeSize], length(posVocab), options[:posSize]),
#     GRU(options[:wordSize] + options[:shapeSize] + options[:posSize], options[:hiddenSize])
# )

prepend!(vocabularies.words, ["UNK"])
wordIndex = Dict{String,Int}(word => i for (i, word) in enumerate(vocabularies.words))
shapeIndex = Dict{String,Int}(shape => i for (i, shape) in enumerate(vocabularies.shapes))
posIndex = Dict{String,Int}(pos => i for (i, pos) in enumerate(vocabularies.partsOfSpeech))
labelIndex = Dict{String,Int}(label => i for (i, label) in enumerate(vocabularies.labels))

Xs, Ys = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)

@info Xs[1]
@info Ys[1]
