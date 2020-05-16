"""
    Vietnamese spelling checker.
    phuonglh@gmail.com
    Feb. 16, 2020
"""

using Flux
using Flux: onehotbatch, onehot, onecold, batchseq, reset!, crossentropy, throttle
using Base.Iterators: partition
using Flux: @epochs
using Statistics: mean
using BSON: @save, @load

inputPath = "dat/txt/vtb.txt"
g = false
N = 5000
hid = 64
maxEpochs = 15

include("Vocab.jl")

sentences = readlines(inputPath)
# convert sentences to lowercase
sentences = removeDelimiters.(lowercase.(sentences))
for sentence in sentences[1:20]
    println(sentence)
end

frequency = vocab(sentences, 1)
lexicon = collect(frequency)
# sort the word by frequency in decreasing order
sort!(lexicon, by = p -> p.second, rev = true)

println("#(lexicon) = ", length(lexicon))
foreach(println, lexicon[1:100])

# build the alphabet of all characters
alphabet = Set([' '])
for syllable in keys(frequency)
    for c in syllable
        push!(alphabet, c)
    end
end
println("#(alphabet) = ", length(alphabet))
if g
    println("GPU")
else
    println("CPU")
end
println(alphabet)

alphabet = collect(alphabet)
# when mutating a sentence, we use an alphabet without space to keep sentence lengths unchanged.
alphabetWithoutSpace = filter(x -> x != ' ', alphabet)
# use two mutation operations that do not change the length of sentences
mutation = [:swap, :replace]

# mutate sentences and write the jumbled strings to an output file
# or read the previouly available data set to reconstruct mutated sentences
generated = false
mutatedSentences = if (generated)
    # generate mutated data set and save to an external file for latter use
    ms = map(s -> mutateSentence(s, alphabetWithoutSpace, mutation, 0.15), sentences)
    file = open(inputPath * ".mutated", "w")
    for sentence in ms
        write(file, join(map(p -> string(p[1])[1], sentence), ' '))
        write(file, "\n")
        write(file, join(map(p -> p[2], sentence), ' '))
        write(file, "\n")
    end
    close(file)
    ms
else
    # load the previously generated data file
    lines = readlines(inputPath * ".mutated")
    ms = Array{Array{Tuple{Symbol,String}},1}()
    i = 1
    while i < length(lines)
        global i
        y = map(a -> Symbol(a), split(lines[i], " "))
        x = string.(split(lines[i+1], " "))
        push!(ms, collect(zip(y, x)))
        i = i + 2
    end
    ms
end

"""
    Transforms a sentence into an array of one-hot vectors based on an alphabet.
    Return a pair of (x, y) in the training mode or a single x in the test mode, where
    x and y are matrices of size length(alphabet) x length(sentence).
"""
function vectorize(x::String, y::String = "", training::Bool = false)
    xs = onehotbatch(x, alphabet)
    if (training)
        ys = onehotbatch(y, alphabet)
        (Float32.(xs), Float32.(ys))
    else 
        Float32.(xs)
    end
end

inp = length(alphabet)
out = length(alphabet)

# Create a model
model = Chain(
    Dense(inp, hid),
    GRU(hid, hid),
    Dense(hid, out),
    softmax
)
println(model)

# use GPU if g is true
if g model = gpu(model) end

# the full model which takes as input a 2-d matrix representing an input 
# sequence; each column corresponds to a character of the sequence.
function f(x)
    reset!(model)
    prediction = model(x)
    return prediction
end

# Loss function on a batch. For each sequence in the batch, 
# we apply the model and compute the cross-entropy element-wise.
# The total loss of the batch is returned.
loss(xb, yb) = sum(crossentropy.(f.(xb), yb))

batchSize = 32
println("Vectorizing the dataset... Please wait.")
# XYs is an array of samples  [(x_1, y_1), (x_2, y_2,),... ]
input = map(a -> join(map(p -> p[2], a), ' '), mutatedSentences)

data = collect(zip(input, sentences))[1:N]
@time XYs = map(s -> vectorize(s[1], s[2], true), data)

# convert a 2-d array to an array of column vectors
flatten(xs) = [xs[:, i] for i = 1:size(xs,2)]
# extracts Xs and Ys
Xs = map(pair -> flatten(pair[1]), XYs)
Ys = map(pair -> flatten(pair[2]), XYs)

# batch a sequence with padding p
batches(xs, p) = [batchseq(b, p) for b in partition(xs, batchSize)]
# batch Xs with a zero vector
Xb = batches(Xs, Float32.(zeros(inp)))
# batch Ys with a zero vector
Yb = batches(Ys, Float32.(zeros(out)))

# bring data to GPU if g is true
if g 
    println("Bring data to GPU...")
    Xb = map(t -> gpu.(t), Xb)
    Yb = map(t -> gpu.(t), Yb)
end

# create a data set for training, each training point is a pair of batch
dataset = collect(zip(Xb, Yb))
println("#(batches) = ", length(dataset))

X1 = Xb[1]
Y1 = Yb[1]
println("typeof(X1) = ", typeof(X1)) # this should be Array{Array{Float32,2},1}
println("typeof(Y1) = ", typeof(Y1)) # this should be Array{Array{Float32,2},1}

# train the model with some number of epochs and save the parameters to a BSON file
function train(numEpochs::Int, modelPath::String)
    optimizer = ADAM(.001)
    evalcb = () -> @show(loss(X1, Y1)) # or use loss(dataset[1]...)
    @epochs numEpochs Flux.train!(loss, params(model), dataset, optimizer, cb = throttle(evalcb, 30))
    theta = Tracker.data.(params(model)) |> cpu    
    @save modelPath theta
end

# Predicts the correct sentence of a mutated sentence
function predict(sentence::String)::String
    reset!(model)
    x = vectorize(sentence)
    y = onecold(model(x) |> cpu)
    result = map(e -> alphabet[e], y)
    join(result)
end

# Predicts a list of sentences, collect prediction result and report prediction accuracy
# xs: mutated sentences; ys: correct sentences
function evaluate(xs::Array{String}, ys::Array{String})::Tuple{Float64,Array{String}}
    zs = map(s -> predict(s), xs)
    goodCorrections = 0
    badCorrections = 0
    correctDifferences = 0
    predictDifferences = 0
    for i = 1:length(ys)
        u, v, w = split(xs[i], " "), split(ys[i], " "), split(zs[i], " ")
        (x, y) = (collect(u), collect(v))
        (y, z) = (collect(v), collect(w))
        xyDiff = x .== y
        # number of zeros in xyDiff is the number of mutations that need to be corrected
        correctDifferences = correctDifferences + (length(y) - sum(xyDiff))
        zyDiff = z .== y
        # number of zeros in zyDiff is the number of changes that the model makes
        predictDifferences = predictDifferences + (length(y) - sum(zyDiff))
        if (length(y) != length(z))
            println("Wrong whitespace prediction, which makes lengths different!")
            println(y)
            println(z)
        end
        # if xyDiff = [0, 1, 1] then there is one mutation at the first token.
        # if zyDiff = [1, 0, 1] then the model correctly recovers the first token
        # but it make a wrong change in the second token, the third token is correctly left unchanged.
        # goodCorrections: the number of zeros in xyDiff which are correctly changed to 1 in zyDiff
        # badCorrections: the number of ones in xyDiff which are incorrectly changed to 0 in zyDiff
        for j=1:length(y)
            if (xyDiff[j] == false) && (zyDiff[j] == true)
                goodCorrections = goodCorrections + 1
            end
            if (xyDiff[j] == true) && (zyDiff[j] == false)
                badCorrections = badCorrections + 1
            end
        end
    end
    recall = goodCorrections / correctDifferences
    precision = goodCorrections / predictDifferences
    fScore = 2*recall*precision/(recall + precision)
    # prediction/correct pairs
    ts = map(p -> string(p[1], "/", p[2]), zip(zs, ys))
    println("correctDifferences = $(correctDifferences)")
    println("predictDifferences = $(predictDifferences)")
    println("goodCorrections = $(goodCorrections)")
    println("badCorrections = $(badCorrections)")
    (100*fScore, ts)
end

# Some constants
prefix = string(homedir(), "/vlp/")
@time train(maxEpochs, string(prefix, inputPath * ".bson"))

(trainScore, trainPrediction) = evaluate(input[1:N], sentences[1:N])
println(trainScore)

foreach(println, trainPrediction[1:10])
