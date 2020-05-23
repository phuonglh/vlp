#=
    Word shape detector.
    phuonglh
    November 8, 2019
=#

struct Pattern
    name::String
    pattern::String
    regexp::Regex
    weight::Int
    Pattern(n, p) = new(n, p, Regex(p), 0)
    Pattern(n, p, w) = new(n, p, Regex(p), w)
end

"An array of bricks, each captures a text pattern."
const patterns = Pattern[
    Pattern("acronym", raw"\b\p{Lu}\p{Ll}\.?\p{Lu}+\.?\b", 5),
    Pattern("email", raw"(\w[-._%:\w]*@\w[-._\w]*\w\.\w{2,3})", 4),
    Pattern("url", raw"(((\w+)\://)+[a-zA-z][\-\w]*\w+(\.\w[\-\w]*)+(/[\w\-]+)*(\.\w+)?(/?)(\?(\w+=[\w%]+))*(&(\w+=[\w%]+))*|[a-z]+((\.)\w+)+)", 3),
    Pattern("name", raw"\b(\p{Lu}\p{Ll}+)([\s+_&\-]?(\p{Lu}\p{Ll}+))+\b", 2),
    Pattern("allcap", raw"\b[\p{Lu}]{2,}\b", 1),
    Pattern("date1", raw"\b(([12][0-9]|3[01]|0*[1-9])[-/.](1[012]|0*[1-9])[-/.](\d{4}|\d{2})|(1[012]|0*[1-9])[-/.]([12][0-9]|3[01]|0*[1-9])[-/.](\d{4}|\d{2}))\b", 1),
    Pattern("date2", raw"\b(1[012]|0*[1-9])[-/](\d{4}|\d{2})\b", 1),
    Pattern("date3", raw"\b([12][0-9]|3[01]|0*[1-9])[-/](1[012]|0*[1-9])\b", 1),
    Pattern("time", raw"\b\d{1,2}:\d{1,2}\b", 1),
    Pattern("numberSeq", raw"\+?\d+(([\s.-]+\d+)){2,}\b", 1),
    Pattern("duration", raw"\b\d{4}\-\d{4}\b", 1),
    Pattern("currency", raw"\p{Sc}+\s?(\d*)?\d+([.,]\d+)*\b"),
    Pattern("number", raw"([+-]?(\d*)?[\d]+([.,]\d+)*%?)"),
    Pattern("item", raw"\d+[.)]\b"),
    Pattern("punct", raw"[-@…–~`'“”’‘|\/$.,:;!?'\u0022]+"),
    Pattern("bracket", raw"[\}\{\]\[><\)\(]+"),
    Pattern("capital", raw"\b\p{Lu}+[\p{Ll}_]*[+]?\b"),
    Pattern("phrase", raw"\b[\p{Ll}\s_]+\b"),
    Pattern("other", raw".+", -1)
]

"Detects the shape of a word."
function shape(word::String)::String
    for brick in patterns
        m = match(brick.regexp, word)
        if m !== nothing
            return brick.name
        end
    end
    return "<unknown>"
end
