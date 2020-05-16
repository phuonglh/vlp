#=
    Implementation of Bidirectional RNN model.
    phuonglh@gmail.com
    November 21, 2019.
=#

using Flux

# a Bidirectional GRU working on input matrix
struct BiGRU
    left
    right
end

BiGRU(inp::Integer, hid::Integer) = BiGRU(GRU(inp, div(hid, 2)), GRU(inp, div(hid, 2)))

# Apply a BiGRU on an input x, which is a matrix of dimension DxN.
function apply(f, x)
    result = vcat(f.left(x), reverse(f.right(reverse(x, dims=2)), dims=2))
    Flux.reset!(f.left)
    Flux.reset!(f.right)
    return result
end

# overload call, so the object can be used as a function
(f::BiGRU)(x) = apply(f, x)

