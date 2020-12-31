# phuonglh

# Compute and show the histogram of length of sentences in a dataset

using Flux

include("../../tdp/src/Oracle.jl")

function histogram(path::String, transition::Bool=false)
    # histogram of sentence length
    sentences = readCorpus(path)
    ls = if (transition) 
        map(sentence -> length(decode(sentence)), sentences)
    else
        map(sentence -> length(sentence.tokens), sentences)
    end
    frequency = Flux.frequencies(ls)
    xs = sort!(collect(keys(frequency)))
    ys = map(x -> frequency[x], xs)
    #plot(xs, ys, xlabel="token length", ylabel="count", legend=false)
    (xs, ys)
end

