
struct Brick
  name::String
  pattern::String
  regexp::Regex
  weight::Int
  Brick(n, p) = new(n, p, Regex(p), 0)
  Brick(n, p, w) = new(n, p, Regex(p), w)
end

"An array of bricks, each captures a text pattern."
bricks = Brick[
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

"""
  shape(word)

  Get the word shape or word form of a word.
"""
function shape(word::String)::String
  for brick in bricks
    m = match(brick.regexp, word) 
    if m !== nothing
      return brick.name
    end
  end
  return "UNK"
end