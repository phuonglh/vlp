#=
    Generate some data for testing BiRNN
    phuonglh@gmail.com
=#

include("BiRNN.jl")

x = rand(3, 5)

m = Chain(
    Dense(3, 4),
    BiGRU(4, 6)
)
y = m(x)

println(y)
println(size(y))

a1 = rand(3, 5)
x1 = [a1[:, j] for j =1:5]

a2 = rand(3, 8)
x2 = [a2[:, j] for j =1:8]

a3 = rand(3, 7)
x3 = [a3[:, j] for j =1:7]

Xs = [x1, x2, x3]
println("Xs = ")
foreach(println, Xs)

batchSize = 2
batches(xs, p) = [Flux.batchseq(b, p) for b in Iterators.partition(xs, batchSize)]
Xb = batches(Xs, zeros(3))
println("Xb = ")
foreach(println, Xb)

