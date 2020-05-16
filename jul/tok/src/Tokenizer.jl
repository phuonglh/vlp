#=
App:
- Julia version: 1.1.0
- Author: phuonglh
- Date: 2019-06-03
=#

#push!(LOAD_PATH, pwd())
#println(LOAD_PATH)

using VietnameseTokenizer

include("Unicode.jl")

"The main function."
function main(verbose::Bool = false)
    println("#(bricks) = $(length(bricks))")
    # sort the bricks by weight in decreasing order
    sort!(bricks, by = b -> b.weight, rev = true)

    s = """Bộ Công an cáo buộc ông Trương Duy Nhất vi phạm pháp luật khi sử dụng nhà đất công sản tại Đà Nẵng, liên quan vụ án của Vũ "Nhôm"."""
    println(s)
    tokens = tokenize(s)
    println(tokens)
    ss = readlines("dat/txt/vlsp.txt")
    println("#(lines) = $(length(ss))")

    println("Sequential processing: ")
    file = open("dat/txt/vlsp.jul.tok", "w")
    @time for s in ss
        tokens = tokenize(convertUTF8(s)) # convert composite Vietnamese text to pre-compose text
        words = map(t -> t.text, tokens)
        write(file, join(words, " "))
        write(file, "\n")
    end
    close(file)
end

main()