
struct Token
  word::String
  annotation::Dict{Symbol,String}
end

struct Sentence
  tokens::Array{Token}
end

"""
  readCorpus(path, maxSentenceLength=40)

  Read a CoNLLU file to build graphs. Each graph is a sentence.
"""
function readCorpus(path::String, maxSentenceLength::Int=40)::Array{Sentence}
  lines = filter(line -> !startswith(line, "#"), readlines(path))
  append!(lines, [""])
  sentences = []
  tokens = []
  for line in lines
    parts = split(strip(line), r"\t+")
    if length(parts) == 1
      prepend!(tokens, [Token("ROOT", Dict(:id => "0", :lemma => "NA", :upos => "NA", :pos => "NA", :fs => "NA", :head => "NA", :label => "NA"))])
      # add sentence if it is not too long...
      if length(tokens) <= maxSentenceLength
        push!(sentences, Sentence(tokens))
      end
      empty!(tokens)
    else
      word = parts[2]
      fullLabel = parts[8]
      colonIndex = findfirst(':', fullLabel)
      label = if (colonIndex !== nothing) fullLabel[1:colonIndex-1] else fullLabel end
      annotation = Dict(:id => parts[1], :lemma => parts[3], :upos => parts[4], :pos => parts[5], :fs => parts[6], :head => parts[7], :label => label)
      push!(tokens, Token(word, annotation))
    end
  end
  sentences
end
