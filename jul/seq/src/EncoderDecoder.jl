# phuonglh@gmail.com
# Implementation of a sequence-to-sequence model in Julia
# using Flux

using Flux


inputSize = 32
hiddenSize = 16

# 1. Create encoder
forwardEncoder = GRU(inputSize, hiddenSize÷2)
backwardEncoder = GRU(inputSize, hiddenSize÷2)

"""
    encode(xs)

    Encode a sequence which is represented by a matrix, where each column corresponds to 
    a vector representation of token in the sequence. We use a bi-directional GRU for encoding.
"""
function encode(xs::Array{Float32,2}) 
    vcat.(forward(xs), Flux.flip(backward, xs))
end

# 2. Create an alignment (or attention) model which scores the degree of match between 
#  an output position and an input position. The attention model that we use here is simply a linear one.
alignNet = Dense(2*hiddenSize, 1)

"""
    align(s, h)

    Align the decoder output `s` with hidden state of inputs `h`.
"""
function align(s::Array{Float32,2}, h::Array{Float32,2})
    alignNet(vcat(h, s .* trues(1, size(h, 2))))
end

