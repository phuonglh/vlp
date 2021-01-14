# phuonglh@gmail.com

using DataFrames
using JSONTables
using JSON3
using Dates

"""
    readDF(path)

    Read objects in a JSONL file into a data frame. 
"""
function readDF(path::String, N::Int=100)::DataFrame
    # read lines from the score path, concatenate them into an json array object
    lines = readlines(path)[1:N]
    # collect all fields
    objects = map(line -> JSON3.read(line), lines)
    ka = union([keys(object) for object in objects]...)
    s = string("[", join(lines, ","), "]")
    # convert to a json table
    jt = jsontable(s)
    data = Dict(Symbol(k) => get.(jt, k, missing) for k in ka)
    # convert to a data frame, if a field is not present then it is marked as `missing`
    DataFrame(data)
end

function selectOrder(df)::DataFrame
    select(df, :u_shpcod => :shop, :u_tmonbi => :amount, :docstatus => :status, :docdate => :date)
end

df = readDF("/home/phuonglh/FRT/data/decryptedData/ordr012018.json", 20)
order = selectOrder(df)
