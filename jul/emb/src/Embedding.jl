#=
    Build an embedding layer for NLP.
    phuonglh@gmail.com
    November 12, 2019, updated on December 9, 2020
=#

using Flux

# Each embedding layer contains a matrix of all word vectors, 
# each column is the vector of the corresponding word index.
struct Embedding
    W
end

Embedding(inp::Int, out::Int) = Embedding(rand(Float32, out, inp))

# overload call, so the object can be used as a function
# x is a word index (1 <= x < vocabSize)
(f::Embedding)(x) = f.W[:,x]

# make the embedding layer trainable
Flux.@functor Embedding

# Embeddings of word, shape and part-of-speech
struct EmbeddingWSP
    word::Embedding
    shape::Embedding
    partOfSpeech::Embedding
end

EmbeddingWSP(inpW::Int, outW::Int, inpS::Int, outS::Int, inpP::Int, outP::Int) = EmbeddingWSP(Embedding(inpW, outW), Embedding(inpS, outS), Embedding(inpP, outP))
# x is an array of 3 indices or a matrix of size 3 x maxSeqLen where each column is an array of 3 indices
# For example, if x is the matrix of two columns: [ 3 4; 2 3; 1 2], then the first column is the vector [3 2 1] which represents
# the first token of word index 3, shape index 2 and part-of-speech index 1; and the second column is the vector [4 3 2] which
#  represents the second token of word index 4, shape index 3 and part-of-speech index 2.
# For each token, we concatenate its word embedding, shape embedding and part-of-speech embedding 
# Note that we use hcat(xs...) instead of Flux.batch(xs) since the back-propagation algorithm does not support mutating array on-the-fly.
(f::EmbeddingWSP)(x) = hcat([vcat(f.word(x[1,t]), f.shape(x[2,t]), f.partOfSpeech(x[3,t])) for t=1:size(x,2)]...)
Flux.@functor EmbeddingWSP

# Example usage: 
# x = [3 4 5; 2 3 4; 1 2 3]
# f = EmbeddingWSP(6,3,4,2,3,1)
# f(x) should give a matrix of size 6x3 (6 = 3 + 2 + 1)
