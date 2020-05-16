#=
    Reads pre-trained word vectors from an external file.
    phuonglh
    November 9, 2019.
=#

function load(path::String)
    println("Loading word vectors, please wait...")
    vectors = Dict{String, Array{Float32}}()
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    for i = 3:length(lines)
        parts = string.(split(lines[i], r"\s+"))
        word = parts[1]
        x = map(token -> parse(Float32, token), parts[2:end])
        vectors[word] = x
    end
    vectors
end
