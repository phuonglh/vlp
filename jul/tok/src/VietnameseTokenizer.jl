#=
VietnameseTokenizer:
- Julia version: 1.1.0
- Author: phuonglh
- Date: 2019-05-27
- Last update: 2019-11-08
=#

module VietnameseTokenizer

export Brick, bricks, lexicon, tokenize

struct Brick
    name::String
    pattern::String
    regexp::Regex
    weight::Int
    Brick(n, p) = new(n, p, Regex(p), 0)
    Brick(n, p, w) = new(n, p, Regex(p), w)
end

"An array of bricks, each captures a text pattern."
const bricks = Brick[
    Brick("acronym", raw"\b\p{Lu}\p{Ll}\.?\p{Lu}+\.?\b", 5),
    Brick("email", raw"(\w[-._%:\w]*@\w[-._\w]*\w\.\w{2,3})", 4),
    Brick("url", raw"(((\w+)\://)+[a-zA-z][\-\w]*\w+(\.\w[\-\w]*)+(/[\w\-]+)*(\.\w+)?(/?)(\?(\w+=[\w%]+))*(&(\w+=[\w%]+))*|[a-z]+((\.)\w+)+)", 3),
    Brick("name", raw"\b(\p{Lu}\p{Ll}+)([\s+_&\-]?(\p{Lu}\p{Ll}+))+\b", 2),
    Brick("allcap", raw"\b[\p{Lu}]{2,}\b", 1),
    Brick("date1", raw"\b(([12][0-9]|3[01]|0*[1-9])[-/.](1[012]|0*[1-9])[-/.](\d{4}|\d{2})|(1[012]|0*[1-9])[-/.]([12][0-9]|3[01]|0*[1-9])[-/.](\d{4}|\d{2}))\b", 1),
    Brick("date2", raw"\b(1[012]|0*[1-9])[-/](\d{4}|\d{2})\b", 1),
    Brick("date3", raw"\b([12][0-9]|3[01]|0*[1-9])[-/](1[012]|0*[1-9])\b", 1),
    Brick("time", raw"\b\d{1,2}:\d{1,2}\b", 1),
    Brick("numberSeq", raw"\+?\d+(([\s.-]+\d+)){2,}\b", 1),
    Brick("duration", raw"\b\d{4}\-\d{4}\b", 1),
    Brick("currency", raw"\p{Sc}+\s?(\d*)?\d+([.,]\d+)*\b"),
    Brick("number", raw"([+-]?(\d*)?[\d]+([.,]\d+)*%?)"),
    Brick("item", raw"\d+[.)]\b"),
    Brick("punct", raw"[-@…–~`'“”’‘|\/$.,:;!?'\u0022]+"),
    Brick("bracket", raw"[\}\{\]\[><\)\(]+"),
    Brick("capital", raw"\b\p{Lu}+[\p{Ll}_]*[+]?\b"),
    Brick("phrase", raw"\b[\p{Ll}\s_]+\b"),
    Brick("other", raw".+", -1)
]

const lexicon = Set{String}(map(w -> lowercase(strip(w)), readlines("dat/lexicon.txt")))

struct Token
    i::Int
    form::String
    text::String
end

"The main function which segments an array of syllables into words."
segment(syllables::Array{String}, forward::Bool = false, verbose::Bool = false) =
    if forward
        segmentForward(syllables, verbose)
    else
        segmentBackward(syllables, verbose)
    end

"Maximal matching from left to right."
function segmentForward(syllables::Array{String}, verbose::Bool = false)::Array{String}
    result = String[]
    token = syllables[1]
    word = token
    if (length(syllables) >= 1)
        n = min(11, length(syllables) - 1)
        m = 1
        for i = 1:n
            token = string(token, ' ', syllables[i+1])
            if token in lexicon
                word = token
                m = i + 1
                if verbose println(word); end
            end
        end
        push!(result, word)
        if m < n
            right = segmentForward(syllables[m+1:end])
            for w in right push!(result, w); end
        end
    end
    result
end

"Maximal matching from right to left."
function segmentBackward(syllables::Array{String}, verbose::Bool = false)::Array{String}
    result = String[]
    token = syllables[end]
    word = token
    if (length(syllables) >= 1)
        n = max(1, length(syllables) - 11)
        m = length(syllables)
        for i=length(syllables)-1:-1:n
            token = string(syllables[i], ' ', token)
            if token in lexicon
                word = token
                m = i
                if verbose println(word); end
            end
        end
        push!(result, word)
        if m > 1
            left = segmentBackward(syllables[1:m-1])
            for w in left push!(result, w); end
        end
    end
    result
end

"The core algorithm for segmenting a sentence into tokens."
function run(s::String, forward::Bool = false, verbose::Bool = false)::Array{Token}
    result = Token[]
    if !isempty(strip(s))
        for brick in bricks
            m = match(brick.regexp, s)
            if m != nothing
                x = Token(m.offset, brick.name, m.match)
                if verbose println("\t$x"); end
                if x.i > 1
                    u = prevind(s, x.i)
                    left = s[1:u]
                    if verbose println("left = \"$(left)\""); end
                    ys = run(left)
                    for y in ys 
                        push!(result, y); 
                    end
                end
                if (x.form == "phrase")
                    syllables = string.(split(x.text, ' '))
                    tokens = segment(syllables, forward, verbose)
                    if !forward reverse!(tokens); end
                    for token in tokens 
                        push!(result, Token(x.i, "word", token)); 
                    end
                else
                    push!(result, x)
                end
                if x.i + lastindex(x.text) <= lastindex(s)
                    v = nextind(s, x.i + lastindex(x.text) - 1)
                    right = s[v:end]
                    if verbose println("right = \"$(right)\""); end
                    ys = run(right)
                    for y in ys 
                        push!(result, y); 
                    end
                end
                # do not consider the second matched brick
                break
            end
        end
    end
    result
end

"Merge capital tokens with their subsequent word tokens if necessary."
function merge(tokens::Array{Token}, verbose::Bool = false)::Array{Token}
    result = Token[]
    i = 1
    n = length(tokens)
    if n >= 2
        # find the first capital token
        while (i <= n && tokens[i].form != "capital") i = i + 1; end
        for j = 1:(i-1) 
            push!(result, tokens[j]); 
        end
        if (i <= n)
            token = tokens[i].text
            w = token
            k = i
            for j = (i+1):min(i+3,n)
                w = string(w, ' ', tokens[j].text)
                if (lowercase(w) in lexicon)
                    k = j
                    token = w
                end
            end
            push!(result, Token(i, "capital", token))
            rest = merge(tokens[k+1:end], verbose)
            for t in rest 
                push!(result, t); 
            end
        end
    else
        if n >= 1
            push!(result, tokens[1])
        end
    end
    result
end

"Segments a sentence into tokens and enumerates the tokens."
function tokenize(s::String, forward::Bool = false, verbose::Bool = false)::Array{Token}
    result = Token[]
    xs = run(s, forward, verbose)
    tokens = merge(xs, verbose)
    for j=1:length(tokens)
        word = replace(tokens[j].text, r"\s+" => "_")
        push!(result, Token(j, tokens[j].form, word))
    end
    result
end

end