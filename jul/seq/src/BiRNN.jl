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

# Apply a BiGRU on an input x of dimension DxN.
apply(f, x) = vcat(f.left(x), reverse(f.right(reverse(x, dims=2)), dims=2)) 

# overload call, so the object can be used as a function
(f::BiGRU)(x) = apply(f, x)

Flux.@functor BiGRU