# phuonglh@gmail.com
# A user-defined layer to accept multiple inputs and output a 
# concatenated vector of some selected vectors. The input is fed in
# as a tuple of 2 components: one token sequence for RNN path and 
# one token sequence for MLP path

using Flux

struct Join
    fs # functions (typically two layers: [EmbeddingWSP, RNN])
end

Join(fs...) = Join(fs)

"""
    x: token id matrix of size (3 x sentenceLength), each column contains 3 ids for (word, shape, tag)
    y: token position vector which corresponds to a parsing configuration
"""
function (g::Join)(x::Tuple{Array{Int,2},Array{Int,1}})
    a, b = x
    as = g.fs[1](a)
    u = g.fs[2](as) # if `fs[2]` is a RNN and `as` is an index array, this gives a sequence
    vec(u[:, b])  # if `b` is an index array, this gives a concatenated vector 
end

function (g::Join)(x::SubArray)
    g(x[1])
end

"""
    x: token id matrix of size (3 x sentenceLength), each column contains 3 ids for (word, shape, tag)
    y: token position matrix in which each column corresponds to a parsing configuration
"""
function (g::Join)(x::Tuple{Array{Int,2},Array{Int,2}})
    a, b = x
    as = g.fs[1](a)
    u = g.fs[2](as) # if `fs[2]` is a RNN and `as` is an index array, this gives a sequence
    vs = [vec(u[:, b[:,j]]) for j=1:size(b,2)] # apply for each column in b
    hcat(vs...) # stack vs to get the output matrix instead of an array of arrays
end


Flux.@functor Join
