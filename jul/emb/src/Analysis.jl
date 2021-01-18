### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ d465bfb6-596d-11eb-3dac-cff61af6a1f6
begin
	using Plots
	using Distances
end

# ╔═╡ f75c9efe-596d-11eb-20fa-cf8f37219a1c
path = string(homedir(), "/vlp/jul/emb/dat/vie/wv.txt")

# ╔═╡ 121879ac-596e-11eb-3ef2-ab9081fbacf9
function readWordVectors(path)
	function makeEntry(line)
		parts = split(line, " ")
		u = string(parts[1])
		v = parse.(Float32, parts[2:end])
		u => v
	end
	lines = readlines(path)
	entries = map(line -> makeEntry(line), lines)
	Dict(entries...)
end

# ╔═╡ 53c01138-5970-11eb-2ac9-833df3939594
wv = readWordVectors(path)

# ╔═╡ 06446b1a-5971-11eb-3ac6-2f6f8e00ed80
pronouns = ["anh", "em", "chị", "bác", "chú", "dì"]

# ╔═╡ 22061748-5971-11eb-3d17-cdd8e9efad54
pvs = map(p -> wv[p], pronouns)

# ╔═╡ 5e4aae7e-5970-11eb-2988-d10cec021c39
function cosineSimilarity(as)
	n = length(as)
	a = zeros(n,n)
	for i = 1:n
		for j = i:n
			a[i,j] = 1 - cosine_dist(wv[as[i]], wv[as[j]])
			a[j,i] = a[i,j]
		end
	end
	return a
end

# ╔═╡ ef59cab8-597e-11eb-1cf6-dd577dbad28b
cosineSimilarity(pronouns)

# ╔═╡ 54477138-597e-11eb-2073-4f3fc208ac83
cities = ["biên_hòa", "bà_rịa", "hà_nam", "hà_nội"]

# ╔═╡ f7a408b6-597e-11eb-2d82-a312144b737a
cosineSimilarity(cities)

# ╔═╡ 33222b74-597e-11eb-0b5d-0924848d60ab
1 - cosine_dist(wv["ban_ngày"], wv["ban_đêm"])

# ╔═╡ 1e399ebc-597f-11eb-04d0-932d2c1c6a38
names = ["nguyễn_thiện_nhân", "nguyễn_thiện_thuật", "nguyễn_thành_phương", "nguyễn_văn_thuật", "nguyễn_văn_linh"]

# ╔═╡ 88343caa-597f-11eb-00eb-0fb7c0d79264
cosineSimilarity(names)

# ╔═╡ 90111bb4-597f-11eb-1d36-25b788a7fdeb


# ╔═╡ Cell order:
# ╠═d465bfb6-596d-11eb-3dac-cff61af6a1f6
# ╠═f75c9efe-596d-11eb-20fa-cf8f37219a1c
# ╠═121879ac-596e-11eb-3ef2-ab9081fbacf9
# ╠═53c01138-5970-11eb-2ac9-833df3939594
# ╠═06446b1a-5971-11eb-3ac6-2f6f8e00ed80
# ╠═22061748-5971-11eb-3d17-cdd8e9efad54
# ╠═5e4aae7e-5970-11eb-2988-d10cec021c39
# ╠═ef59cab8-597e-11eb-1cf6-dd577dbad28b
# ╠═54477138-597e-11eb-2073-4f3fc208ac83
# ╠═f7a408b6-597e-11eb-2d82-a312144b737a
# ╠═33222b74-597e-11eb-0b5d-0924848d60ab
# ╠═1e399ebc-597f-11eb-04d0-932d2c1c6a38
# ╠═88343caa-597f-11eb-00eb-0fb7c0d79264
# ╠═90111bb4-597f-11eb-1d36-25b788a7fdeb
