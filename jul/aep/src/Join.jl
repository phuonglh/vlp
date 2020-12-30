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

function (g::Join)(x::Tuple{Array{Int,2},Array{Int,1}})
    a, b = x
    u = g.fs[2](g.fs[1](a)) # if `fs[2]` is a RNN and `a` is an index array, this gives a sequence
    vec(u[:, b])  # if `b` is an index array, this gives a concatenated vector 
end

function (g::Join)(x::SubArray)
    g(x[1])
end

Flux.@functor Join
