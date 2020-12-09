
using Flux

include("Vocab.jl")
include("Oracle.jl")

options = Dict{Symbol,Any}(
    :minFreq => 2,
    :lowercase => true,
    :numFeatures => 16536, 
    :corpusPath => string(pwd(), "/jul/tdp/dat/tests.conllu"),
    :modelPath => string(pwd(), "/jul/tdp/dat/")
)

"""
    train(options)

    Train a neural network transition classifier.
"""
function train(options::Dict{Symbol,Any})
    sentences = readCorpus(options[:corpusPath])
    contexts = collect(Iterators.flatten(map(sentence -> decode(sentence), sentences)))
    @info "Number of sentences = $(length(sentences))"
    @info "Number of contexts  = $(length(contexts))"
    vocabulary, labels = vocab(contexts, options[:minFreq], options[:lowercase])
    foreach(println, labels)
end

train(options)