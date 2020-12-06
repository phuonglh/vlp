using DataStructures

include("Sentence.jl")

struct Arc
  head::String
  dependent::String
  label::String
end

mutable struct Config
  sentence::Sentence
  stack::Stack{String}()
  queue::Queue{String}()
  arcs::Array{Arc}
end

"""
  next(config, transition)

  Find the next config given the current config and a transition. The transition is a string of format 
  <symbol>[-<label>], for example: "SH", "RA-dobj", "LA-nsubj", etc.
"""
function next(config::Config, transition::String)::Config
  if transition == "SH"
    push!(config.stack, dequeue!(config.queue))
  else if transition == "RE"
    u = first(config.stack)
    # pre-condition: there exists an arc (v, u).
    condition = !isempty(filter(arc -> arc.dependent == u, config.arcs))
    if (condition)
      pop!(config.stack)
    else
      @warn "Pre-condition for RE is not satisfied!"
    end
  else if startswith(transition, "LA")
    u = pop!(config.stack)
    v = first(config.queue)
    # pre-condition: there does not exist an arc (k, u)
    condition = isempty(filter(arc -> arc.dependent == u, config.arcs))
    if condition
      push!(config.arcs, Arc(v, u, transition[4:end]))
    else
      @warn "Pre-condition for LA is not satisfied!"
    end
  else if startswith(transition, "RA")
    u = first(config.stack)
    v = dequeue!(config.queue)
    # pre-condition: there does not exist an arc (k, v)
    condition = isempty(filter(arc -> arc.dependent == v, config.arcs))
    if condition
      push!(config.stack, v)
      push!(config.arcs, Arc(u, v, transition[4:end]))
    else
      @warn "Pre-condition for RA is not satisfied!"
    end
  end
  Config(config.sentence, config.stack, config.queue, config.arcs)
end

"""
  decode(sentence)

  Decode an annotated sentence (or a dependency graph) to get a sequence of (currentConfig, nextAction) pairs.
"""
function decode(sentence::Sentence)::Array{(Config, String)}
  σ = Stack{String}()
  β = Queue{String}()
  push!(β, Token("ROOT", Dict(:id => "0")))
  push!(β, sentence.tokens...)
  A = Array{Arc}()
  config = Config(sentence, σ, β, A)
  pairs = Array{(Config, String)}()
  while !isempty(β)
    push!(pairs, (config, "SH"))
    config = next(config, "SH")
  end
  pairs
end