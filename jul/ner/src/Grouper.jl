#=
    Groups an array of integers by their identity. This utility function 
    is used to create same-length batches of a dataset.
    phuonglh
    November 10, 2019.
=#

function group(xs::Array{Int})::Array{Array{Int}}
    n = length(xs)
    pairs = collect(zip(1:n, xs))
    sort!(pairs, by = p -> p[2])
    is = map(p -> p[1], pairs)
    ys = Array{Array{Int},1}()
    i = 1
    j = i+1
    while j <= n
        while j <= n && pairs[j][2] == pairs[i][2]
            j = j + 1
        end
        push!(ys, is[i:j-1])
        i = j
        j = i + 1
        if (i == n)
            push!(ys, [is[i]])
        end
    end
    return ys
end
