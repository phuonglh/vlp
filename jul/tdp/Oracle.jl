using DataStructures

include("Sentence.jl")

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
    if !reducible(config)
      @warn "Pre-condition for RE is not satisfied!"
    end
    pop!(config.stack)
  elseif startswith(transition, "LA")
    u = first(config.stack)
    # pre-condition: there does not exist an arc (k, u)
    condition = isempty(filter(arc -> arc.dependent == u, config.arcs))
    if !condition
      @warn "Pre-condition for LA is not satisfied!"
    end
    u = pop!(config.stack)
    v = first(config.queue)
    push!(config.arcs, Arc(v, u, transition[4:end]))
  elseif startswith(transition, "RA")
    v = first(config.queue)
    # pre-condition: there does not exist an arc (k, v)
    condition = isempty(filter(arc -> arc.dependent == v, config.arcs))
    if !condition
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
  decode(sentence)

  Decode an annotated sentence (or a dependency graph) to get a sequence of (currentConfig, nextAction) pairs.
  Here, we use the ARC-EAGER parsing method.
"""
function decode(sentence::Sentence)::Array{Tuple{Config,String}}
  σ = Stack{String}()
  β = Queue{String}()
  for id in map(token -> token.annotation[:id], sentence.tokens)
    enqueue!(β, id)
  end
  A = Array{Arc,1}()
  arcList = map(token -> ((token.annotation[:head], token.annotation[:id]), token.annotation[:label]), sentence.tokens)
  arcMap = Dict{Tuple{String, String}, String}(arcList)

  foreach(println, arcList)

  config = Config(σ, β, A)
  pairs = []
  push!(pairs, (config, "SH"))
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
    @info string(config, " => ", transition)
    push!(pairs, (config, transition))
    config = next(config, transition)
  end
  pairs
end

sentences = readCorpus("jul/tdp/dat/tests.conllu")
pairs = decode(sentences[1])
foreach(println, pairs)

# In most practical parser implementations, an incomplete graph is converted into a tree by adding arcs from the root node to all words that lack a head.