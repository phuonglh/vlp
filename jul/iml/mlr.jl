# X of size NxD, y of size Nx1, θ of size KxD
function cost(X::Array{Float64,2}, y::Array{Int}, θ::Array{Float64,2}, λ::Float64=0.)::Float64
  u = sum(θ[y, :] .* X, dims=2) # col vector
  v = sum(exp.(θ*X'), dims=1)   # row vector
  t = θ[:,2:end]
  J = -sum(u) + sum(log.(v)) 
  J/length(y) + λ*sum(t .* t)/2
end

function gradient(X, y, θ, λ=0.)::Array{Float64,2}
  K, D = size(θ)
  G = zeros(K, D)
  P = exp.(θ*X')
  z = sum(P, dims=1)
  for k in 1:K
      p = P[k,:] ./ z'
      δ = (y .== k)
      r = [0 ; θ[k,2:end]]
      G[k,:] = X'*(p - δ)/length(y) + λ*r
  end
  G
end

function g!(∇, θ)
  G = gradient(X, y, θ)
  K, D = size(θ)
  for k in 1:K
    ∇[k,:] = G[k,:]
  end
end

function classify(X, θ)::Array{Int}
	scores = θ*X'
	indices = argmax(scores, dims=1)[:]
	map(i -> (indices[i])[1], 1:N)
end

# main program
#

using DelimitedFiles, Optim

A = readdlm("dat/jul/iris-train.txt", '\t')
X = A[:, 2:5]
y = Int.(A[:, 1])
N, D = size(X)
K = length(unique(y))

θ = zeros(K, D)
J0 = cost(X, y, θ)
println("J(0) = ", J0)

result_nm = optimize(t -> cost(X, y, t), θ, NelderMead(), Optim.Options(iterations = 2000))
θ_nm = Optim.minimizer(result_nm)
println("θ_nm: ")
println(θ_nm)
println("J(θ_nm) = ", cost(X, y, θ_nm))


result_bfgs = optimize(t -> cost(X, y, t), g!, θ)
θ_bfgs = Optim.minimizer(result_bfgs)
println("θ_bfgs: ")
println(θ_bfgs)
println("J(θ_bfgs) = ", cost(X, y, θ_bfgs))

ŷ = classify(X, θ_bfgs)
accuracy = sum(ŷ .== y)/N
println("training accuracy = ", accuracy)