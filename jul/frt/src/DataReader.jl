# phuonglh@gmail.com

using DataFrames
using JSONTables
using JSON3
using Dates
using FLoops
using BangBang
using MicroCollections
using CSV

# All fields of orderDF are Set([:docstatus, :u_buslin, :type_returnso, :u_tmonpr, :u_exdate, :doctype, :u_tmonbi, :docdate, :docentry, :u_tmontx, :u_shpcod, :u_desc])
# We are interested in 5 selected orders.
orderFields = Dict(:u_shpcod => :shop, :u_tmonbi => :amount, :docstatus => :status, :docdate => :date, :docentry => :doc, :doctype => :type)

"""
    readOrderDF(path, numRecords)

    Read objects in an order JSONL file into a data frame. 
"""
function readOrderDF(path::String, numRecords::Int=-1)::DataFrame
    # read lines from the score path, concatenate them into an json array object
    lines = readlines(path)
    xs = if (numRecords > 0) 
        lines[1:numRecords] 
    else
        lines
    end
    # collect all objects and fields
    @info "Reading order objects from $(path)..."
    objects = map(line -> JSON3.read(line), xs)
    @info "Joining strings"
    s = string("[", join(xs, ","), "]");
    @info "Converting to a json table..."
    jt = jsontable(s)
    @info "Preparing columns. Create entries for selected fields in parallel, using 4 CPU cores.."
    executor = ThreadedEx(basesize = length(orderFields)÷4)
    @time @floop executor for field in keys(orderFields)
        @reduce(entries = append!!(EmptyVector(), [orderFields[field] => get.(jt, field, missing)]))
        entries
    end
    data = Dict(entries)
    # convert to a data frame, if a field is not present then it is marked as `missing`
    @info "Preparing df..."
    df = DataFrame(data)
    # transform the df
    dropmissing!(df)
    ef = select(df, 
        :shop => :shop,
        :amount => (x -> parse.(Float32, x) ./ 10^6) => :amount, 
        :status => :status,
        :date => (x -> parse.(Date, map(d -> d[1:10], x))) => :date,
        :doc => :doc,
        :type => :type
    )
    return ef
end

# Selected fields of an order detail data frame.
orderDetailFields = Dict(:itemcode => :item, :dscription => :desc, :quantity => :qty, :discprcnt => :disc, 
    :u_tmoney => :price, :whscode => :shop, :docentry => :doc)
# it turns out that :disc is not in percentage unit, it is the same unit as :price.

"""
    readOrderDetailDF(path, numRecords)

    Read objects in an order detail JSONL file into a data frame. 
"""
function readOrderDetailDF(path::String, numRecords::Int=-1)::DataFrame
    # read lines from the score path, concatenate them into an json array object
    lines = readlines(path)
    xs = if (numRecords > 0) 
        lines[1:numRecords] 
    else
        lines
    end
    # collect all objects and fields
    @info "Reading order detail objects from $(path)..."
    objects = map(line -> JSON3.read(line), xs)
    @info "Joining strings"
    s = string("[", join(xs, ","), "]");
    @info "Converting to a json table..."
    jt = jsontable(s)
    @info "Preparing columns. Create entries for selected fields in parallel, using 4 CPU cores.."
    executor = ThreadedEx(basesize = length(orderDetailFields)÷4)
    @time @floop executor for field in keys(orderDetailFields)
        @reduce(entries = append!!(EmptyVector(), [orderDetailFields[field] => get.(jt, field, missing)]))
        entries
    end
    data = Dict(entries)
    # convert to a data frame, if a field is not present then it is marked as `missing`
    @info "Preparing df..."
    df = DataFrame(data)
    # transform the df
    dropmissing!(df)
    # remove all rows which does not have a good (8-character length) shop code or start with 1100 (warehouses aren't treated)
    withValidShop = df[length.(df.shop) .== 8 .| startswith.(df.shop, "1100"), :] 
    ef = select(withValidShop, 
        :item => :item,
        :desc => :desc,
        :doc => :doc,
        :disc => (x -> parse.(Float32, x) ./ 10^6) => :disc,
        :shop => (x -> map(v -> v[1:5], x)) => :shop,
        :price => (x -> parse.(Float32, x) ./ 10^6) => :price, 
        :qty => (x -> Int.(parse.(Float32, x))) => :qty
    )
    return ef
end

"""
    preprocess(year)

    Read all order data and order detail data into data frames, preprocess them and write
    backs the DFs to JSON files for later processing.
"""
function preprocess(year::String="2018")
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    for month in months
        input = string(pwd(), "/dat/frt/ordr", month, year, ".json")
        orderDF = readOrderDF(input)
        output = string(pwd(), "/jul/frt/dat/o", year, month, ".csv")
        CSV.write(output, orderDF)

        input = string(pwd(), "/dat/frt/rdr1", month, year, ".json")
        detailDF = readOrderDetailDF(input)
        output = string(pwd(), "/jul/frt/dat/r", year, month, ".csv")
        CSV.write(output, detailDF)
    end
end
