# phuonglh@gmail.com
# Sentence encoders

using Flux

include("Embedding.jl")
include("Options.jl")

encoder = Chain(
    Embedding(options[:numFeatures], options[:embeddingSize]),
    GRU(options[:embeddingSize], options[:hiddenSize])
)
