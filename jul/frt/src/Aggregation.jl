# phuonglh@gmail.com

using Statistics
using DataFrame
using CSV

include("Options.jl")

"""
    aggregateSale(options, u, v, year)

    Aggregate sale information of an SKU (item) at a given shop from month `u` to month `v` of a year.
"""
function aggregateSale(options, u=1, v=12, year="2018")
	monthSt(month) = if month < 10 string("0", month); else string(month); end
	folder = options[:folder]
	oPaths = [string(folder, "o", year, monthSt(m), ".csv") for m in u:v]
	rPaths = [string(folder, "r", year, monthSt(m), ".csv") for m in u:v]
	sales = []
	for m = u:v
		# load and process order df, extract transactions of the interested shop
		odf = CSV.File(oPaths[m]) |> DataFrame 
		df = odf[(odf.status .== "F") .& (odf.shop .!= 11000) .& (odf.shop .!= 11001), :]
		gdf = groupby(df, :shop)
		of = gdf[(options[:shop],)]
		order = select(of, [:doc, :date, :status])
		# load and process order detail df
		rdf = CSV.File(rPaths[m]) |> DataFrame
	    ef = rdf[rdf.price .> options[:minPrice], :]
	    gef = groupby(ef, :item) 
		rf = gef[(options[:item],)]
		detail = select(rf, [:doc, :disc, :price, :qty])
		# join two data frames
		jdf = innerjoin(order, detail, on = :doc)
		# group by date and compute the total quantity for each date
		byDate = groupby(jdf, :date)
		sale = combine(byDate, :price => mean => :price, :disc => mean => :disc, :qty => sum => :quantity)
	    sort!(sale, :date)
		push!(sales, sale)
	end
	# stack the data frame together 
	return vcat(sales...)
end

"""
    aggregate(xs, width,=1 agg=sum)

    Aggregation of a series using an aggration method (summation, average, maximum, minimum).
    Default method is summation.
"""
function aggregate(xs, width=1, agg=sum)
    ys = collect(Iterators.partition(xs, width))
    map(y -> agg(y), ys)
end


function createPredictor()
end

# Forecasting models: 
# - Linear space model: ARIMA or ETS
# - Bayesian structural time series model: Prophet
# - Non-parametric time series: NPTS (Amazon)
# - DL methods: DeepAR+: train a single model jointly over the entire collection of the time series in the dataset 
#    (grouping of demand for different products, server loads, requests for web pages)

# Target: allows missing values
# Features: price, weekday, weekend, holidays, promotion. Not allow missing values.
# CSV: (item SKU, timestamp, target values)
# CSV: (item SKU, timestamp, features)
# Categorical related data: static information: color of items, binary indicator (e.g., smart or not smart)
# Local model: one model per SKU


# ORDR

# - docentry: so don hang
# - doctype: loai don hang (01: Ban tai quay; 02: Ban tra gop;...)
# - docstatus: trang thai (A: Đã duyệt, C: Hủy, D: Hoàn tất thu cọc, F: Hoàn tất, N: Chờ xử lý, O: Mở, S: Lưu, T: Đã trả hàng, W: Đang trả hàng)
# - ubus_lin: kenh ban hang
# - u_shpcod: cua hang
# - u_tmonpr: tong tien chua thue
# - u_exdate: ngay doi hang
# - u_desc: ghi chu
# - u_tmontx: tong tien thue
# - u_tmonbi: tong tien phai thu
# - type_returnso: kieu don hang
# - docdate: ngay ban


# RDR1

# - itemcode: ma san pham
# - dscription: ten san pham
# - quantity: so luong
# - discprcnt: chiet khau
# - U_TMoney: gia ban
# - Whscode: ma kho (shop+local department)
# - docentry: so don hang ==> to link with ORDR df
