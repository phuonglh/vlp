"""
    Introduce spelling errors to a word by
    - swapping two adjacent characters, or
    - replacing one character with a different character, or
    - inserting one character, or
    - deleting one character
"""

using Random
Random.seed!(220712)


delimiters = r"[\s,.:;!*+?)()\"“”’‘\u0022\u200b\ufeff\u200e\uf0b7\x7f\b]+"


"""
    validIndices(s)

    Gets valid indices of a Vietnamese string. For example, 
    "đường phố" => [1, 3, 5, 8, 9, 10, 11, 12, 13]
"""
function validIndices(s::String)::Array{Int}
    j = 0
    indices = Array{Int,1}()
    while j < lastindex(s)
        j = nextind(s, j)
        push!(indices, j)
    end
    indices
end

"""
    mutate(s, alphabet, mutation)

    Mutate a word according to an allowed mutation types.
    INP: a string to mutate and an alphabet
    OUT: a pair of mutation type and the mutated string
"""
function mutate(s::String, alphabet::Array{Char}, mutation::Array{Symbol}=[:swap, :replace, :insert, :delete])::Tuple{Symbol,String}
    i = validIndices(s)
    m = length(i)
    if (m < 2) return (:none, s) end
    r = rand(1:length(mutation), 1)[1]
    j = rand(1:m, 1)[1]
    if (r == 1)
        if (m >= 2)
            rest = (j + 2 <= m) ? s[i[j+2]:i[m]] : ""
            if (j == 1)
                return (:swap, string(s[i[j+1]], s[i[j]], rest))
            elseif (j == m) 
                begin
                    left = (m > 2) ? s[1:i[m-2]] : ""
                    return (:swap, string(left, s[i[j]], s[i[j-1]]))
                end
            else
                return (:swap, string(s[1:i[j-1]], s[i[j+1]], s[i[j]], rest))
            end
        end
    elseif (r == 2)
        u = rand(1:length(alphabet), 1)[1]
        c = alphabet[u]
        left = (j > 1) ? s[1:i[j-1]] : ""
        right = (j < m) ? s[i[j+1]:i[m]] : ""
        return (:replace, string(left, c, right))
    elseif (r == 3)
        u = rand(1:length(alphabet), 1)[1]
        c = alphabet[u]
        left = s[1:i[j]]
        right = (j < m) ? s[i[j+1]:i[m]] : ""
        return (:insert, string(left, c, right))
    elseif (r == 4)
        left = (j > 1) ? s[1:i[j-1]] : ""
        right = (j < m) ? s[i[j+1]:i[m]] : ""
        return (:delete, string(left, right))
    end
end

"""
    mutate(tokens, alphabet, mutation, β)

    Mutates a sentence in the form of an array of tokens (syllables). Each token is randomly 
    mutated with a probability of β. Returns the mutated sentence.

    INP: [tôi, đang ăn, cơm, tối]
    OUT: [tối/:replace, đng/:delete, ănp/:insert, cơm/:none, tiố/:swap]
"""
function mutateSentence(tokens::Array{String}, alphabet::Array{Char}, mutation::Array{Symbol}=[:swap, :replace, :insert, :delete], β::Float64 = 0.1)::Array{Tuple{Symbol,String}}
    f(x::String) = (rand() <= β) ? mutate(x, alphabet, mutation) : (:none, x)
    map(token -> f(token), tokens)
end

"""
    mutateSentence(s, alphabet, mutation, β)

    Mutates a sentence in the form of a string. First, we segment the sentence into 
    an array of syllables using whitespace characters. Next, we mutate that array. 
"""
function mutateSentence(s::String, alphabet::Array{Char}, mutation::Array{Symbol}=[:swap, :replace, :insert, :delete], β::Float64 = 0.1)::Array{Tuple{Symbol,String}}
    x = filter(token -> length(strip(token)) > 0, split(s, delimiters))
    mutateSentence(string.(x), alphabet, mutation, β)
end

function removeDelimiters(s::String)::String
    xs = split(s, delimiters)
    strip(join(xs, ' '))
end

"""
    generate(sentences, outputPath, alphabet, mutation, β)

    Generate mutated versions of given sentences and write results to an output file. The default `mutation` operation is either swap 
    or replace which keep the length of the sentences the same before and after mutation.
"""
function generate(sentences::Array{String}, outputPath::String, alphabet::Array{Char}, mutation::Array{Symbol} = [:swap, :replace], β::Float64 = 0.1)
    # generate mutated data set and save to an external file for latter use
    ms = map(s -> mutateSentence(s, alphabet, mutation, β), sentences)
    file = open(outputPath, "w")
    for sentence in ms
        write(file, join(map(p -> string(p[1])[1], sentence), ' '))
        write(file, "\n")
        write(file, join(map(p -> p[2], sentence), ' '))
        write(file, "\n")
    end
    close(file)
end


function test()
    alphabet = ['β', 'α', 'γ', 'ζ', 'φ']
    tokens = ["những", "ngày", "xưa", "thân", "ái", "em", "gửi", "lại", "cho", "tôi", "!"]

    mutation = [:swap, :replace]
    xs = mutateSentence(tokens, alphabet, mutation, 0.3)
    foreach(println, xs)
    ys = join(map(x -> x[2], xs), " ")
    println(ys)

    xs = mutateSentence("Đường thương đau đầy ải nhân gian, ai chưa qua", alphabet, mutation, 0.3)
    foreach(println, xs)
    ys = join(map(x -> x[2], xs), " ")
    println(ys)
end

function generate(path::String)
    sentences = lowercase.(readlines(path))
    mutation = [:swap, :replace, :insert, :delete]
    alphabet = ['%', '&', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '@', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă', 'đ', 'ĩ', 'ũ', 'ơ', 'ư', 'ạ', 'ả', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ụ', 'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ', '–']
    generate(sentences, string(path, ".inp"), alphabet, mutation)
end
