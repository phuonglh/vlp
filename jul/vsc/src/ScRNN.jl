using Flux
using Flux: onehotbatch, onehot, onecold, batchseq, reset!, crossentropy, throttle
using Base.Iterators: partition
using Flux: @epochs
using Statistics: mean
using BSON: @save, @load

using Random
Random.seed!(220712)

# Implementation of the sermi-character RNN model for spelling check.
# phuonglh@gmail.com

include("Vocab.jl")
include("Options.jl")

# 0. load mutated sentences from a data file which was previously generated
# and build an array of sequences. Each sequence is an array of pairs: [(label_1, word_1), (label_2, word_2)...].
# This will be used as labeled sequences for learning.
function readData(options)
  lines = readlines(options[:inputPath])
  mutatedSentences = Array{Array{Tuple{Symbol,String}},1}()
  i = 1
  while i < length(lines)
      y = map(a -> Symbol(a), split(lines[i], " "))
      x = string.(split(lines[i+1], " "))
      push!(mutatedSentences, collect(zip(y, x)))
      i = i + 2
  end
  return mutatedSentences
end

mutatedSentences = readData(options)
input = map(a -> map(p -> p[2], a), mutatedSentences)
output = map(a -> map(p -> p[1], a), mutatedSentences)

# 1. build syllable frequency and an alphabet
sentences = map(s -> join(s, " "), input)
frequency = vocab(sentences, options[:minFrequency])
lexicon = collect(frequency)
# sort the word by frequency in decreasing order
sort!(lexicon, by = p -> p.second, rev = true)
@info "First 10 tokens: $(join(lexicon[1:10], ", "))"
alphabet = Set{Char}()
for syllable in keys(frequency)
  for c in syllable
      push!(alphabet, c)
  end
end
alphabet = collect(alphabet)
@info "#(alphabet) = $(length(alphabet))"
@info join(alphabet, " ")

# 2. featurize the data set.

"""
  boc(s, alphabet)

  Compute bag-of-character vectors for middle characters of a string, that is s[2:end-1].
"""
function boc(s::String, alphabet::Array{Char})
    a = onehotbatch(collect(s[nextind(s, 1):prevind(s, lastindex(s))]), alphabet)
    sum(a, dims=2)
end

"""
  vectorize(x, y, training=false)

  Transforms a sentence into an array of vectors based on an alphabet and 
  a label set. Return a pair of (x, y) in the training mode or a single x in the test mode, where
  x and y are matrices of size length(lexicon) x length(sentence).
"""
function vectorize(x::Array{String}, y::Array{Symbol} = Array{Symbol,1}(), training::Bool = false)
  # one-hot first char vector
  ucs = map(token -> first(token), x)
  us = onehotbatch(ucs, alphabet)
  # one-hot last char vector
  vcs = map(token -> last(token), x)
  vs = onehotbatch(vcs, alphabet)
  # one-hot middle bag-of-character vector
  cs = zeros(length(alphabet), length(x))
  for j = 1:length(x)
      cs[:,j] = boc(x[j], alphabet)
  end
  # combine all vectors into xs and convert xs to Float32 to speed up computation
  xs = Float32.(vcat(us, vs, cs))
  if (training)
      ys = onehotbatch(y, options[:labels])
      (Float32.(xs), Float32.(ys))
  else 
      Float32.(xs)
  end
end

# """
#   f(x)

#   Apply the model on an input sentence in the form of a 2-d matrix, each column corresponds to a token of the sequence.
# """
# function f(x)
#     reset!(model)
#     prediction = model(x)
#     return prediction
# end

# 4. vectorize the data set and create mini-batches of data
println("Vectorizing the dataset... Please wait.")
# XYs is an array of samples  [(x_1, y_1), (x_2, y_2,),... ]
data = collect(zip(input, output))
@time XYs = map(s -> vectorize(s[1], s[2], true), data)

# convert a 2-d array to an array of column vectors
flatten(xs) = [xs[:, j] for j = 1:size(xs,2)]
# extracts Xs and Ys
Xs = map(pair -> flatten(pair[1]), XYs)
Ys = map(pair -> flatten(pair[2]), XYs)

# batch a sequence with padding p
batches(xs, p) = [batchseq(b, p) for b in partition(xs, options[:batchSize])]
# batch Xs, pad with zero vectors
Xb = batches(Xs, Float32.(zeros(3*length(alphabet))))
# batch Ys, pad with zero vectors
Yb = batches(Ys, Float32.(zeros(length(options[:labels]))))

if options[:gpu] 
  println("Bringing data to GPU...")
  Xb = map(t -> gpu.(t), Xb)
  Yb = map(t -> gpu.(t), Yb)
end

# create a data set for training, each training point is a pair of batch
dataset = collect(zip(Xb, Yb))
@info "#(batches) = $(length(dataset))"

X1 = Xb[1]
Y1 = Yb[1]
@info "typeof(X1) = $(typeof(X1))" # this should be Array{Array{Float32,2},1}
@info "typeof(Y1) = $(typeof(Y1))" # this should be Array{Array{Float32,2},1}

model = Chain(
  Dense(3*length(alphabet), options[:embeddingSize]), 
  GRU(options[:embeddingSize], options[:hiddenSize]), 
  Dense(options[:hiddenSize], length(options[:labels])),
  softmax
)
@info model  

"""
  predict(sentence)

  Predict the correct sentence of a mutated sentence. This is always done on CPU 
"""
function predict(sentence::Array{String})::Array{Symbol}
  reset!(model)
  x = vectorize(sentence)
  y = onecold(model(x))
  map(e -> options[:labels][e], y)
end

"""
  evaluate(xs, ys)

  Predict a list of sentences, collect prediction result and report prediction accuracy
  xs: mutated sentences; ys: correct mutation labels
"""
function evaluate(xs::Array{Array{String,1}}, ys::Array{Array{Symbol,1}})::Tuple{Dict{Symbol,Float64},Array{String}}
  zs = map(s -> predict(s), xs)
  good = Dict{Symbol,Int}() # number of correct predictions for each label
  bad = Dict{Symbol,Int}() # number of incorrect predictions for each label
  foreach(k -> good[k] = 0, options[:labels])
  foreach(k -> bad[k] = 0, options[:labels])
  for i = 1:length(ys)    
    y, z = ys[i], zs[i]
    zyDiff = z .== y
    for i = 1:length(y)
      k = y[i]
      if (zyDiff[i])
        good[k] = good[k] + 1
      else 
        bad[k] = bad[k] + 1
      end
    end
  end
  accuracy = Dict{Symbol,Float64}()
  for k in options[:labels]
    accuracy[k] = good[k]/(good[k] + bad[k])
  end
  @info "\tgood dictionary = $(good)"
  @info "\t bad dictionary = $(bad)"
  # prediction/correct pairs
  ts = map(p -> string(p[1], "/", p[2]), zip(zs, xs))
  (accuracy, ts)
end

function train(options)
  function cost(x, y) # cost on a sample sequence
    c = crossentropy(model(x), y)
    reset!(model)
    return c
  end
  # loss on a batch
  loss(xb, yb) = sum(cost(xb[i], yb[i]) for i=1:length(xb))

  optimizer = ADAM()

  function evalcb() 
    @info "loss = $(loss(X1, Y1))" # or use loss(dataset[1]...)
    a, _ = evaluate(input, output)
    @info "\ttraining accuracy = $a"
  end
  global model
  @epochs options[:numEpochs] Flux.train!(loss, params(model), dataset, optimizer, cb = throttle(evalcb, 60))
  # save the model to a BSON file
  if (options[:gpu])
      model = model |> cpu
  end
  @save options[:modelPath] model
  return model
end


# hidden = [100, 500, 550, 600, 700, 800, 900, 1024]
# outputFile = open(options[:outputPath], append=true)
# for i=1:length(hidden)
#   global hid = hidden[i]
#   global model = Chain(Dense(inp, hid), LSTM(hid, hid), Dense(hid, out), softmax)
#   if options[:gpu] model = gpu(model) end
#   write(outputFile, string(model))
#   write(outputFile, "\n")
#   train(options)
#   if (options[:gpu])
#     global model = cpu(model)
#   end
#   (trainScore, trainPrediction) = evaluate(input[1:N], output[1:N])
#   write(outputFile, string(trainScore))
#   write(outputFile, "\n\n")
#   flush(outputFile)
# end
# close(outputFile)

train(options)
@info "Evaluating the model..."
a, _ = evaluate(input, output)
@info a
