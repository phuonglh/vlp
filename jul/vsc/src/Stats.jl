#=
Stats:
- Julia version: 
- Author: phuonglh
- Date: 2020-03-15
=#

xs = readlines("dat/vsc/vtb-mutated.txt")
is = filter(i -> i % 2 == 1, collect(1:length(xs)))
ys = xs[is]
println(length(ys))
as = map(y -> map(label -> label[1], split(y, ' ')), ys)
frequency = Dict{Char,Int}('n' => 0, 's' => 0, 'r' => 0, 'd' => 0, 'i' => 0)
for a in as
    for word in a
        frequency[word] += 1
    end
end
println(frequency)
total = sum(values(frequency))
println("total = ", total)
for key in keys(frequency)
    println("$(key): $(frequency[key]/total*100)")
end