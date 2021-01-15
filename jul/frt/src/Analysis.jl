# phuonglh@gmail.com
using StatsPlots

include("DataReader.jl")


"""
    analyseOrder(df)

    Analyse the sale by date of the top shop. `df` is a monthly ORDR data frame as provided by FRT, 
    for example January 2018. Return a data frame and top shop code.
"""
function analyseOrder(df)
    @info "Analysing the order df..."
    # select the orders with status "F" (complete); there are 347,258 complete records in January 2018
    # filter out 2 warehouses (11000 and 11001)
    complete = order[(order.status .== "F") .& (order.shop .!= "11000") .& (order.shop .!= "11001"), :]
    # group the complete df by shop, there are 516 shops in January 2018
    gdf = groupby(complete, :shop)
    # count the number of orders for each shop 
    cs = [(i, size(gdf[i], 1)) for i = 1:length(gdf)]
    # sort by number of orders in descending order
    sort!(cs, by = p -> p[2], rev = true)
    @assert length(cs) >= 10
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
    return (sdf, topShop[1,1])
end

function analyseOrder(path::String)
    df = readOrderDF(string(pwd(), path), -1)  
    analyseOrder(df)
end

function analyseOrderDetail(df)
    # remove all rows having a missing value
    ef = dropmissing!(df)
    # remove all rows which does not have a good (8-character length) shop code or start with 1100 (warehouses)
    withValidShop = ef[length.(ef.shop) .== 8 .| startswith.(ef.shop, "1100"), :] 
    # select and transform columns
    ff = select(withValidShop, 
        :entry => :entry,
        :item => :item,
        :shop => (x -> map(v -> v[1:5], x)) => :shop,
        :amount => (x -> parse.(Float32, x) ./ 10^6) => :amount, 
        :quantity => (x -> Int.(parse.(Float32, x))) => :quantity
    )
    # remove all rows whose amount are less than 1
    gf = ff[ff.amount .> 1.0, :]
end

# (sdf, bestShop) = analyseOrder("/dat/frt/ordr052018.json")
# daynames = map(x -> x[1:3], sdf[:,:dayname])
# @df sdf plot(:day, :sale, title="Daily Sale", label=false, xlabel="day of month", ylabel="VND [million]", xticks=(1:length(daynames), daynames), xrotation=90, tickfontsize=6)
