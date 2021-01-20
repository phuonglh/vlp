#=
    Build an embedding layer for NLP.
    phuonglh@gmail.com
    November 12, 2019
=#

using Flux

# Each embedding layer contains a matrix of all word vectors, 
# each column is the vector of the corresponding word index.
struct Embedding
    W
end

# Random embeddings layer
Embedding(vocabSize::Int, outputSize::Int) = Embedding(rand(outputSize, vocabSize))

# overload call, so the object can be used as a function
# x is a word index or an array of word indices (1 <= x < vocabSize)
(f::Embedding)(x) = (f.W)[:, x]

# make the embedding layer trainable
# updated on December 10, 2020
Flux.@functor Embedding