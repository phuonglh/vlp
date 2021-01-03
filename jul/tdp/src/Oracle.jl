using DataStructures

include("Options.jl")
include("Sentence.jl")
include("../../tok/src/Brick.jl")

struct Arc
  head::String
  dependent::String
  label::String
end

struct Config
  stack::Stack{String}
  queue::Queue{String}
  arcs::Array{Arc}
end

struct Context
  features::Array{String}
  transition::String
end


"""
  reducible(config)

  Check whether a config is reducible or not.
"""
function reducible(config::Config)::Bool
  u = first(config.stack)
  # pre-condition: there exists an arc (v, u) where v is not the dummy ROOT with id "0"
  !isempty(filter(arc -> arc.dependent == u && arc.head != "0", config.arcs))
end


"""
  next(config, transition)

  Find the next config given the current config and a transition. The transition is a string of format 
  <symbol>[-<label>], for example: "SH", "RA-dobj", "LA-nsubj", etc.
"""
function next(config::Config, transition::String)::Config
  if transition == "SH"
    push!(config.stack, dequeue!(config.queue))
  elseif transition == "RE"
    if options[:verbose] && !reducible(config)
      @warn "Pre-condition for RE is not satisfied!"
    end
    pop!(config.stack)
  elseif startswith(transition, "LA")
    u = first(config.stack)
    # pre-condition: there does not exist an arc (k, u)
    condition = isempty(filter(arc -> arc.dependent == u, config.arcs))
    if options[:verbose] && !condition
      @warn "Pre-condition for LA is not satisfied!"
    end
    u = pop!(config.stack)
    v = first(config.queue)
    push!(config.arcs, Arc(v, u, transition[4:end]))
  elseif startswith(transition, "RA")
    v = first(config.queue)
    # pre-condition: there does not exist an arc (k, v)
    condition = isempty(filter(arc -> arc.dependent == v, config.arcs))
    if options[:verbose] && !condition
      @warn "Pre-condition for RA is not satisfied!"
    end
    u = first(config.stack)
    v = dequeue!(config.queue)
    push!(config.stack, v)
    push!(config.arcs, Arc(u, v, transition[4:end]))
  end
  Config(config.stack, config.queue, copy(config.arcs))
end

"""
  featurize(config, tokenMap)

  Extract feature strings from a parsing `config`. The `tokenMap` contains a map of tokenId to token.

"""
function featurize(config::Config, tokenMap::Dict{String,Token})::Array{String}
  features = []
  σ, β = config.stack, config.queue
  # top tokens of the stack and queue
  u, v = tokenMap[first(σ)], tokenMap[first(β)]
  s = shape(u.word) 
  push!(features, string("ss0:", s))
  push!(features, string("ws0:", lowercase(u.word)))
  push!(features, string("ls0:", get(u.annotation, :lemma, "NA")))
  push!(features, string("ts0:", get(u.annotation, :pos, "NA")))
  push!(features, string("us0:", get(u.annotation, :upos, "NA")))

  s = shape(v.word)
  push!(features, string("sq0:", s))
  push!(features, string("wq0:", lowercase(v.word)))
  push!(features, string("lq0:", get(v.annotation, :lemma, "NA")))
  push!(features, string("tq0:", get(v.annotation, :pos, "NA")))
  push!(features, string("uq0:", get(v.annotation, :upos, "NA")))
  # second token of the queue
  if length(β) > 1
    id = collect(β)[2]
    v = tokenMap[id]
    s = shape(v.word)
    push!(features, string("sq1:", s))
    push!(features, string("wq1:", lowercase(v.word)))
    push!(features, string("lq1:", get(v.annotation, :lemma, "NA")))
    push!(features, string("tq1:", get(v.annotation, :pos, "NA")))
    push!(features, string("uq1:", get(v.annotation, :upos, "NA")))
  else
    append!(features, ["sq1:[pad]"; "wq1:[pad]"; "lq1:[pad]"; "tq1:[pad]"; "uq1:[pad]"])
  end
  # second token of the stack
  if length(σ) > 1
    id = collect(σ)[2]
    v = tokenMap[id]
    s = shape(v.word)
    push!(features, string("ss1:", s))
    push!(features, string("ws1:", lowercase(v.word)))
    push!(features, string("ls1:", get(v.annotation, :lemma, "NA")))
    push!(features, string("ts1:", get(v.annotation, :pos, "NA")))
    push!(features, string("us1:", get(v.annotation, :upos, "NA")))
  else 
    append!(features, ["ss1:[pad]"; "ws1:[pad]"; "ls1:[pad]"; "ts1:[pad]"; "us1:[pad]"])
  end
  features
end

"""
  decode(sentence)

  Decode an annotated sentence (or a dependency graph) to get a sequence of (currentConfig, nextAction) pairs.
  Then a feature extractor will extract feature strings from the currentConfig.
  Here, we use the ARC-EAGER parsing method. The result will be used as training data for building a transition 
  classifier.
"""
function decode(sentence::Sentence)::Array{Context}
  σ = Stack{String}()
  β = Queue{String}()
  tokenMap = Dict{String,Token}(token.annotation[:id] => token for token in sentence.tokens)
  for id in map(token -> token.annotation[:id], sentence.tokens)
    enqueue!(β, id)
  end
  A = Array{Arc,1}()
  arcList = map(token -> ((token.annotation[:head], token.annotation[:id]), token.annotation[:label]), sentence.tokens)
  arcMap = Dict{Tuple{String, String}, String}(arcList)

  config = Config(σ, β, A)
  contexts = []
  config = next(config, "SH")
  transition = "SH"
  while !isempty(β)
    u, v = first(σ), first(β)
    labelLeft = get(arcMap, (v, u), "NA")
    labelRight = get(arcMap, (u, v), "NA")
    if labelLeft != "NA" # has arc (v, u), extract a LA-label relation
      push!(A, Arc(v, u, labelLeft))
      transition = string("LA-", labelLeft)
    elseif labelRight != "NA" # has arc (u, v), extract a RA-label relation
      push!(A, Arc(u, v, labelRight))
      transition = string("RA-", labelRight)
    elseif reducible(config)
      transition = "RE"
    else
      transition = "SH"
    end
    # @info string(config, " => ", transition)
    push!(contexts, Context(featurize(config, tokenMap), transition))
    config = next(config, transition)
  end
  contexts
end

