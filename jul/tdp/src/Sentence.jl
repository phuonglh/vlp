
struct Token
  word::String
  annotation::Dict{Symbol,String}
end

struct Sentence
  tokens::Array{Token}
end

"""
  readCorpus(path)

  Read a CoNLLU file to build graphs. Each graph is a sentence.
"""
function readCorpus(path::String)::Array{Sentence}
  lines = filter(line -> !startswith(line, "#"), readlines(path))
  append!(lines, [""])
  sentences = []
  tokens = []
  for line in lines
    parts = split(strip(line), r"\s+")
    if length(parts) == 1
      prepend!(tokens, [Token("ROOT", Dict(:id => "0", :head => "NA", :label => "NA"))])
      push!(sentences, Sentence(tokens))
      empty!(tokens)
    else
      word = parts[2]
      annotation = Dict(:id => parts[1], :lemma => parts[3], :upos => parts[4], :pos => parts[5], :fs => parts[6], :head => parts[7], :label => parts[8])
      push!(tokens, Token(word, annotation))
    end
  end
  sentences
end
