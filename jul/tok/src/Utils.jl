#=
Utils:
- Julia version: 1.1.0
- Author: phuonglh
- Date: 2019-06-03
- This tool provides some utility functions.
=#

function numVisibleChars(sentence::String, tokens::Array{Tuple{Int, String, String}})
    # compute the number of visible characters in the input sentence
    num_visible_chars = length(filter(c -> c != ' ', sentence))
    result = tokenize(sentence)
    # compute the number of visible characters in the output result
    c = reduce((a, b) -> a + b, map(token -> length(filter(c -> c != ' ', token[3])), result))
    # the two number of visible characters should be the same, otherwise there is some problem.
    @assert num_visible_chars == c
end

