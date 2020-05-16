"""
  phuonglh
  Train/test split
"""

using Random

"""
    Split a dataset into k folds, each fold contains a pair of 
    (test set, training set).
"""
function folds(k::Int, dataset::Array{Int}, seed::Int = 220712)::Array{Tuple{Array{Int}, Array{Int}}}
    n = length(dataset)
    Random.seed!(seed)
    xs = dataset[randperm(n)]
    u = n ÷ k
    pairs = Tuple{Array{Int}, Array{Int}}[]
    for j = 0:k-1
        α, β = j*u + 1, min((j+1)*u, n)
        test = dataset[α:β]
        train = Array{Int,1}()
        push!(train, dataset[1:α-1]...)
        push!(train, dataset[β+1:n]...)
        push!(pairs, (test, train))
    end
    pairs
end

