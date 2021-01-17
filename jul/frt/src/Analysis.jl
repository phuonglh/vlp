# phuonglh@gmail.com

using DataFrames
using CSV
using StatsPlots
using Statistics


"""
    analyseOrder(df)

    Analyse the sale by date of the top shop. `df` is a monthly ORDR data frame as provided by FRT, 
    for example January 2018. Return a data frame and top shop code.
"""
function analyseOrder(df)
    @info "Analysing the order df..."
    # select the orders with status "F" (complete); there are 347,258 complete records in January 2018
    # filter out 2 warehouses (11000 and 11001)
    complete = df[(df.status .== "F") .& (df.shop .!= 11000) .& (df.shop .!= 11001), :]
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
    # group by date
    efByDate = groupby(topShop, :date)
    # compute the total amount for each date
    sale = combine(efByDate, :amount => sum => :sale)
    sort!(sale, :date)
    sdf = select(sale, :date => (x -> Dates.day.(x)) => :day, :date => (x -> Dates.dayname.(x)) => :dayname, :sale)
    return (topShop, sdf, topShop[1,1])
end

"""
    analyseOrderDetail(df, minPrice)

    Analyse the sale of two top items given order detail information of a month `df`.
"""
function analyseOrderDetail(df, minPrice = 5.)
    # remove all rows whose amount are less than minAmount (milion VND)
    ef = df[df.price .> minPrice, :]
    # group rows by item 
    gf = groupby(ef, :item) 
    # count the number of sale for each item
    cs = [(i, sum(gf[i][:,:qty])) for i = 1:length(gf)]
    # sort by number of orders in descending order
    sort!(cs, by = p -> p[2], rev = true)
    top5 = gf[map(p -> p[1], cs[1:5])]
end

function analyse(month::String="01", year::String="2018")
    orderPath = string(pwd(), "/jul/frt/dat/o", year, month, ".csv")
    orderDF = CSV.File(orderPath) |> DataFrame
    ef, sdf, topShop = analyseOrder(orderDF)

    detailPath = string(pwd(), "/jul/frt/dat/r", year, month, ".csv")
    detailDF = CSV.File(detailPath) |> DataFrame
    top5 = analyseOrderDetail(detailDF)

    ff = select(ef, [:doc, :date, :status])
    # join with third-rank item
    gf = innerjoin(ff, top5[3], on = :doc)
    # group by date
    byDate = groupby(gf, :date)
    # compute the total quantity for each date
    sale = combine(byDate, :price => mean => :price, :disc => mean => :disc, :qty => sum => :quantity)
    sort!(sale, :date)
    return (topShop, top5[3][1,1], sale)
end

# (sdf, bestShop) = analyseOrder("/dat/frt/ordr052018.json")
# daynames = map(x -> x[1:3], sdf[:,:dayname])
# @df sdf plot(:day, :sale, title="Daily Sale", label=false, xlabel="day of month", ylabel="VND [million]", xticks=(1:length(daynames), daynames), xrotation=90, tickfontsize=6)
