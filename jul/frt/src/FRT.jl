### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 12800616-58b1-11eb-0765-8755fbbec375
begin
	using CSV
	using DataFrames
	using Dates
	using Statistics
	using StatsPlots
end

# ╔═╡ e6802b8c-58af-11eb-1c22-07737dede955
include("Analysis.jl")

# ╔═╡ 402c7278-58b0-11eb-1112-01ec4fe5fc24
begin
	month = "01"
	year = "2018"
	orderPath = string(homedir(), "/vlp/jul/frt/dat/o", year, month, ".csv")
	detailPath = string(homedir(), "/vlp/jul/frt/dat/r", year, month, ".csv")
end

# ╔═╡ eff6894e-58b0-11eb-07b6-f543a3a40070
orderDF = CSV.File(orderPath) |> DataFrame

# ╔═╡ 09846520-58b1-11eb-0291-9562d4277107
ef, sdf, topShop = analyseOrder(orderDF)

# ╔═╡ 584d2a88-58b2-11eb-171f-1f894f7d8094
detailDF = CSV.File(detailPath) |> DataFrame

# ╔═╡ a4dc41ea-58b2-11eb-1fa9-2f74905c862a
top5 = analyseOrderDetail(detailDF)

# ╔═╡ 76861e56-58b2-11eb-177d-93c79bc93693
ff = select(ef, [:doc, :date, :status])

# ╔═╡ 0a07b1f8-58b3-11eb-3211-f1d09dd93b16
top5[3]

# ╔═╡ 1d7fad14-58b3-11eb-2c52-edbeda33f152
gf = innerjoin(ff, top5[3], on = :doc)

# ╔═╡ 540415e2-58b3-11eb-31fc-d758fc2c330b
byDate = groupby(gf, :date)

# ╔═╡ 23e32c9c-58b3-11eb-240f-9bcefab941b4
sale = combine(byDate, :price => mean => :price, :disc => mean => :disc, :qty => sum => :quantity)

# ╔═╡ 5025122a-58b3-11eb-2eaa-afc117119a91
daynames = map(x -> x[1:3], sdf[:,:dayname])

# ╔═╡ eba51baa-58b3-11eb-2947-b99adb648140
@df sdf plot(:day, :sale, title="Daily Sale", label=false, xlabel="day of month", ylabel="VND [million]", xticks=(1:length(daynames), daynames), xrotation=90, tickfontsize=6)

# ╔═╡ ea623208-58b5-11eb-2f8b-dd76920303e1
options = Dict{Symbol,Any}(
	:folder => string(homedir(), "/vlp/jul/frt/dat/"),
	:shop => 30220,
	:item => "00395581",
	:minPrice => 5.
)

# ╔═╡ 04d167f0-58b4-11eb-299a-7f364a7a999f
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
	return vcat(sales...)
end

# ╔═╡ 1c71c1c4-58b5-11eb-0f18-8d89e5a281ad
yearSale = aggregateSale(options, 1, 12)

# ╔═╡ 54c25d12-58bb-11eb-2dde-3148ec9ec977
output = string(options[:folder], options[:shop], "-", options[:item], "-2018.csv")

# ╔═╡ c71ebec8-58bb-11eb-073d-87f72e585897
CSV.write(output, yearSale)

# ╔═╡ 4e665c30-58bf-11eb-3672-c5a787e430f5
@df yearSale plot(:date, :quantity, xlabel="date", ylabel="sale", label=options[:item])

# ╔═╡ 7a755718-58bf-11eb-01ec-13e99d5dc9f8
sum(yearSale[!,:quantity])

# ╔═╡ 6c0edd90-58cc-11eb-03b3-e9e83da41dc6
as = transform(yearSale, :date => (x -> Dates.dayname.(x)) => :dayname, :date => (x -> Dates.week.(x)) => :week, :quantity)

# ╔═╡ c03ed92e-58cc-11eb-07e9-01e3d611ac2b
# group the xs data frame by week
byWeek = groupby(as, :week)

# ╔═╡ 6e7519bc-58ce-11eb-39ee-dbfd95bf13c0
weeklySale = combine(byWeek, :price => mean => :price, :disc => mean => :disc, :quantity => sum => :quantity)


# ╔═╡ c4eb6a8a-58ce-11eb-17d6-7741ca557014
@df weeklySale plot(:week, :quantity, xlabel="week", ylabel="weekly quantity", label="SKU: "*options[:item], ytick=1:2:30)

# ╔═╡ 1cf4b02a-5946-11eb-1a1e-7f763228e499
ys = weeklySale[:, :quantity]

# ╔═╡ ab36f946-594a-11eb-0a9a-9fcd529057bc
# simple prediction method
us = ys[1:end-1]

# ╔═╡ e127aa70-594a-11eb-18c2-75bc021ea2ff
simpleError = mean(abs.(us - ys[2:end]))

# ╔═╡ 075bcd02-594b-11eb-1499-37c967014b35
md"""
## Statistics of top 10 SKU 
"""

# ╔═╡ Cell order:
# ╠═12800616-58b1-11eb-0765-8755fbbec375
# ╠═e6802b8c-58af-11eb-1c22-07737dede955
# ╠═402c7278-58b0-11eb-1112-01ec4fe5fc24
# ╠═eff6894e-58b0-11eb-07b6-f543a3a40070
# ╠═09846520-58b1-11eb-0291-9562d4277107
# ╠═584d2a88-58b2-11eb-171f-1f894f7d8094
# ╠═a4dc41ea-58b2-11eb-1fa9-2f74905c862a
# ╠═76861e56-58b2-11eb-177d-93c79bc93693
# ╠═0a07b1f8-58b3-11eb-3211-f1d09dd93b16
# ╠═1d7fad14-58b3-11eb-2c52-edbeda33f152
# ╠═540415e2-58b3-11eb-31fc-d758fc2c330b
# ╠═23e32c9c-58b3-11eb-240f-9bcefab941b4
# ╠═5025122a-58b3-11eb-2eaa-afc117119a91
# ╠═eba51baa-58b3-11eb-2947-b99adb648140
# ╠═ea623208-58b5-11eb-2f8b-dd76920303e1
# ╠═04d167f0-58b4-11eb-299a-7f364a7a999f
# ╠═1c71c1c4-58b5-11eb-0f18-8d89e5a281ad
# ╠═54c25d12-58bb-11eb-2dde-3148ec9ec977
# ╠═c71ebec8-58bb-11eb-073d-87f72e585897
# ╠═4e665c30-58bf-11eb-3672-c5a787e430f5
# ╠═7a755718-58bf-11eb-01ec-13e99d5dc9f8
# ╠═6c0edd90-58cc-11eb-03b3-e9e83da41dc6
# ╠═c03ed92e-58cc-11eb-07e9-01e3d611ac2b
# ╠═6e7519bc-58ce-11eb-39ee-dbfd95bf13c0
# ╠═c4eb6a8a-58ce-11eb-17d6-7741ca557014
# ╠═1cf4b02a-5946-11eb-1a1e-7f763228e499
# ╠═ab36f946-594a-11eb-0a9a-9fcd529057bc
# ╠═e127aa70-594a-11eb-18c2-75bc021ea2ff
# ╠═075bcd02-594b-11eb-1499-37c967014b35
