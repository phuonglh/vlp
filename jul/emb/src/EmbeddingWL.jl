# phuonglh
# Embedding of word (W) and dependency label (L)

using LinearAlgebra

include("Embedding.jl")

struct EmbeddingWL
    word::Embedding
    label::Embedding
end

# words and labels are projected into the same space of `out` dimensions
EmbeddingWL(inpW::Int, inpL::Int, out::Int,) = EmbeddingWL(Embedding(inpW, out), Embedding(inpL, out))


# x is a matrix of size `3 x batchSize where each column contains 3 indices representing head, tail and label
# This function computes a matrix of size `out x batchSize`.
(f::EmbeddingWL)(x) = begin
    # normalize the embedding vectors of the words and 
    # compute the difference vector between `head + label` and `tail`.
    g(h::Int, t::Int, l::Int) = f.word(h)/norm(f.word(h)) + f.label(l) - f.word(t)/norm(f.word(t))
    hcat([g(x[:,t]...) for t=1:size(x,2)]...)
end

Flux.@functor EmbeddingWL
