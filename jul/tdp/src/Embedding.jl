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
# x is a word index or an array, or a matrix (a batch of column vectors) of word indices (1 <= x < vocabSize)
# If x is a matrix, the sum returns a 3-d tensor and we need to squeeze (reshape) the result 
# to get a 2-d matrix.
(f::Embedding)(x) = reshape(sum(f.W[:, x], dims=2), size(f.W,1), size(x,2))

# make the embedding layer trainable
Flux.@functor Embedding