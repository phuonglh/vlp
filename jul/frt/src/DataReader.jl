# phuonglh@gmail.com

using DataFrames
using JSONTables
using JSON3
using Dates
using FLoops
using BangBang
using MicroCollections

# All fields of orderDF are Set([:docstatus, :u_buslin, :type_returnso, :u_tmonpr, :u_exdate, :doctype, :u_tmonbi, :docdate, :docentry, :u_tmontx, :u_shpcod, :u_desc])
# We are interested in a number of selected orders.
orderFields = Dict(:u_shpcod => :shop, :u_tmonbi => :amount, :docstatus => :status, :docdate => :date, :docentry => :entry)

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
    @info "Reading objects..."
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
    DataFrame(data)
end

# Selected fields of an order detail data frame.
orderDetailFields = Dict(:itemcode => :item, :dscription => :desc, :quantity => :quantity, :discprcnt => :discount, 
    :u_tmoney => :amount, :whscode => :shop, :docentry => :entry)

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
    @info "Reading objects..."
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
    DataFrame(data)
end
