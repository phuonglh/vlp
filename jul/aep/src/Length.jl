# phuonglh

# Compute and show the histogram of length of sentences in a dataset

using Flux
using Plots

include("../../tdp/src/Oracle.jl")

# histogram of sentence length
sentences = readCorpus(path)
ls = map(sentence -> length(sentence.tokens), sentences)
frequency = Flux.frequencies(ls)
xs = sort!(collect(keys(frequency)))
ys = map(x -> frequency[x], xs)
plot(xs, ys, xlabel="token length", ylabel="count", legend=false)

# histogram of config sequences
ls = map(sentence -> length(decode(sentence)), sentences)
frequency = Flux.frequencies(ls)
us = sort!(collect(keys(frequency)))
vs = map(u -> frequency[u], us)
plot(us, vs, xlabel="context length", ylabel="count", legend=false)
