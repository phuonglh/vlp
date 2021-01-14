# phuonglh@gmail.com

using DataFrames
using JSONTables
using JSON3
using Dates
using StatsPlots

"""
    readDF(path, numRecords)

    Read objects in a JSONL file into a data frame. 
"""
function readDF(path::String, numRecords::Int=-1)::DataFrame
    # read lines from the score path, concatenate them into an json array object
    lines = readlines(path)
    xs = if (numRecords > 0) 
        lines[1:numRecords] 
    else
        lines
    end
    # collect all objects and fields
    @info "Reading objects..."
    objects = map(line -> JSON3.read(line), xs)
    @info "Find keys..."
    ka = union([keys(object) for object in objects]...)
    @info "Joining strings"
    s = string("[", join(xs, ","), "]")
    # convert to a json table
    jt = jsontable(s)
    @info "Preparing data..."
    data = Dict(Symbol(k) => get.(jt, k, missing) for k in ka)
    # convert to a data frame, if a field is not present then it is marked as `missing`
    @info "Preparing df..."
    DataFrame(data)
end

function selectFields(df)::DataFrame
    select(df, :u_shpcod => :shop, :u_tmonbi => :amount, :docstatus => :status, :docdate => :date)
end

"""
    analyse(df)

    Analyse the sale by date of the top shop. `df` is a monthly ORDR data frame as provided by FRT, 
    for example January 2018.
"""
function analyse(df)
    # select the orders with status "F" (complete); there are 347,258 complete records in January 2018
    # filter out 2 warehouses (11000 and 11001)
    complete = order[(order.status .== "F") .& (order.shop .!= "11000") .& (order.shop .!= "11001"), :]
    # group the complete df by shop, there are 516 shops in January 2018
    gdf = groupby(complete, :shop)
    # count the number of orders for each shop 
    cs = [(i, size(gdf[i], 1)) for i = 1:length(gdf)]
    # sort by number of orders in descending order
    sort!(cs, by = p -> p[2], rev = true)
    # take the top 10 indices
    topTen = map(p -> p[1], cs[1:10])

    # analyse the top-performing shop
    topShop = gdf[first(topTen)]
    # transform the `amount` column to numeric type and `date` column to date type
    ef = select(topShop, 
        :amount => (x -> parse.(Float32, x) ./ 10^6) => :amount, 
        :date => (x -> parse.(Date, map(d -> d[1:10], x))) => :date
    )
    # group ef by date
    efByDate = groupby(ef, :date)
    # compute the total amount for each day
    sale = combine(efByDate, :amount => sum => :sale)
    sort!(sale, :date)
    sdf = select(sale, :date => (x -> Dates.day.(x)) => :day, :date => (x -> Dates.dayname.(x)) => :dayname, :sale)
    return sdf
end

function run(path)
    df = readDF(string(pwd(), path), -1)
    order = selectFields(df)
    analyse(order)
end

# sdf = run("/dat/frt/ordr012018.json")
# @df sdf plot(:day, :sale, title="Sale", label=false, xlabel="day of month", ylabel="VND [million]", xticks=1:31)


#  (78, 2673) => 30220 ==> best-performing shop 
#  (120, 2293) => 30807
#  (130, 2214) => 30242
#  (82, 2142) => 30229 
#  (135, 2062) => 30828

# 31×3 DataFrame
#  Row │ day    dayname    sale    
#      │ Int64  String     Float32 
# ─────┼───────────────────────────
#    1 │     1  Monday     329.364
#    2 │     2  Tuesday    420.757
#    3 │     3  Wednesday  370.325
#    4 │     4  Thursday   430.461
#    5 │     5  Friday     487.044
#    6 │     6  Saturday   448.633
#    7 │     7  Sunday     426.149
#    8 │     8  Monday     564.591
#    9 │     9  Tuesday    358.094
#   10 │    10  Wednesday  599.665
#   11 │    11  Thursday   545.224
#   12 │    12  Friday     422.117
#   13 │    13  Saturday   480.867
#   14 │    14  Sunday     412.07
#   15 │    15  Monday     495.228
#   16 │    16  Tuesday    409.794
#   17 │    17  Wednesday  302.968
#   18 │    18  Thursday   394.247
#   19 │    19  Friday     591.705
#   20 │    20  Saturday   459.546
#   21 │    21  Sunday     475.157
#   22 │    22  Monday     733.137
#   23 │    23  Tuesday    451.962
#   24 │    24  Wednesday  478.069
#   25 │    25  Thursday   550.7
#   26 │    26  Friday     378.652
#   27 │    27  Saturday   339.017
#   28 │    28  Sunday     515.025
#   29 │    29  Monday     627.644
#   30 │    30  Tuesday    409.515
#   31 │    31  Wednesday  435.856