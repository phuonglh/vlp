# Implementation of TransE method
# phuonglh@gmail.com

using Flux
using Flux: @epochs

include("Sampling.jl")
include("EmbeddingWL.jl")

opts = Dict{Symbol,Any}(
    :vocabSize => 2^16,
    :embeddingSize => 16,
    :unknown => "[unk]",
    :numEpochs => 20,
    :wordPath => string(pwd(), "/jul/emb/dat/vie/wv.txt"),
    :labelPath => string(pwd(), "/jul/emb/dat/vie/lv.txt"),
)

words, triplets = extractTriplets(options)
push!(words, opts[:unknown])
labels = unique(map(t -> t.label, collect(triplets)))
vocabSize = min(opts[:vocabSize], length(words))
labelSize = length(labels)

model = EmbeddingWL(vocabSize, labelSize, opts[:embeddingSize])


wordIndex = Dict{String,Int}(element => i for (i, element) in enumerate(words))
labelIndex = Dict{String,Int}(element => i for (i, element) in enumerate(labels))

index(t::Triplet) = [get(wordIndex, t.head, wordIndex[opts[:unknown]]), get(wordIndex, t.tail, wordIndex[opts[:unknown]]), labelIndex[t.label]]


"""
    corruptBatch(X)

    Corrupt a batch of triplets and return a corrupted batch. Note that this returns the same number of triplets (typically the batch size).
"""
function corruptBatch(X::Array{Triplet})::Array{Int,2}
    sample = Array{Triplet,1}()
    for triplet in X
        head = triplet.head
        tail = triplet.tail
        found = false
        # throw a coin to corrupt either head or tail
        while !found
            γ = rand()
            if (γ < 0.5)
                s = Triplet(rand(words), tail, triplet.label)
                if s ∉ triplets
                    push!(sample, s)
                    found = true
                end
            else
                s = Triplet(head, rand(words), triplet.label)
                if s ∉ triplets
                    push!(sample, s)
                    found = true
                end
            end
        end
    end
    Flux.batch([index(t) for t ∈ sample])
end

"""
    batch(xs)

    Create mini-batches of data for training the model, return two arrays of mini-batches corresponding to 
    good and corrupted samples.
"""
function batch(xs::Array{Triplet})::Tuple{Array{Array{Int,2}},Array{Array{Int,2}}}
    Xs = Iterators.partition(xs, options[:batchSize])
    Xb = map(xs -> Flux.batch([index(t) for t ∈ xs]), Xs)
    Yb = map(X -> corruptBatch(collect(X)), Xs)
    (collect(Xb), collect(Yb))
end


"""
    loss(X, Y)

    Define the loss function of the model on a minibatch pair X and Y, both are index matrices of size `3 x batchSize`).
    The loss are the difference between sum of good distances and sum of corrupted distances. L-2 distance is used.
    We will minimize this loss.
"""
function loss(X, Y)
    U = model(X) # matrix of size `out x batchSize`
    V = model(Y) # matrix of size `out x batchSize`
    sum(norm(U[:, j]) for j=1:size(U,2)) - sum(norm(V[:, j]) for j=1:size(V,2))
end

Xs, Ys = batch(collect(triplets))
dataset = collect(zip(Xs, Ys))

evalcb = function()
    weightW = sum(model.word.W)
    weightL = sum(model.label.W)
    @info "Total weight of word embeddings = $(weightW), of label embeddings = $(weightL)"
end

# train the embedding model
@epochs opts[:numEpochs] Flux.train!(loss, params(model), dataset, ADAM(), cb = Flux.throttle(evalcb, 30))

# save the words embeddings and label embeddings to external files
file = open(opts[:wordPath], "w")
for word in words
    v = model.word.W[:, wordIndex[word]]
    write(file, string(word, " ", join(v, " "), "\n"))
end
close(file)

file = open(opts[:labelPath], "w")
for label in labels
    v = model.label.W[:, labelIndex[label]]
    write(file, string(label, " ", join(v, " "), "\n"))
end
close(file)
