using Plots; # plotly()
using DelimitedFiles

# validation
pathValid = "/Users/phuonglh/vlp/vcb/bin/VNM-valid/part-00000-93aea379-12aa-48cc-a1ae-451d8a403bfb-c000.csv"
A = readdlm(pathValid, ',')
y = A[:,1]
z = A[:,2]
x = 1:length(y)
plot(x, [y, z], label=["real" "prediction"], xlabel="time", ylabel="price", legend=:bottomright, title="VNM-valid")
# savefig("bin/VNM-valid.png")

# training
pathTrain = "/Users/phuonglh/vlp/vcb/bin/VNM-train/part-00000-409e11ac-5537-4297-aae8-e33c32ddae99-c000.csv"
B = readdlm(pathTrain, ',')
yt = B[:,1]
zt = B[:,2]
xt = 1:length(yt)
plot(xt, [yt, zt], label=["real" "prediction"], xlabel="time", ylabel="price", legend=:bottomright, title="VNM-train")
# savefig("bin/VNM-train.png")
