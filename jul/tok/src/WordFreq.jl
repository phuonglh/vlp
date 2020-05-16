#=
WordFreq:
- Julia version: 
- Author: phuonglh
- Date: 2019-05-23
=#

# read in a text file
st = read("dat/txt/vlsp.txt", String)
# replace non-alphabet characters from text with a space
nonAlpha = r"(\W\s?)"
st = replace(st, nonAlpha => ' ')
digits = r"(\d+)"
st = replace(st, digits => ' ')
# split text in words
words = split(st, ' ')
# create a frequency table
wordFreq = Dict{String, Int64}()
for word in words
    word = lowercase(strip(word))
    if !isempty(word)
        haskey(wordFreq, word) ? wordFreq[word] += 1 : wordFreq[word] = 1
    end
end
# sort the word by frequency
println("Result:")
words = sort!(collect(keys(wordFreq)))
for word in words
    println("$word: $(wordFreq[word])")
end