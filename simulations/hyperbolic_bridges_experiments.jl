wd = @__DIR__
cd(wd)

using Pkg
Pkg.activate()
include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")
using Random
using ProgressMeter
using StaticArrays
using DelimitedFiles


outdir = "/Users/marc/Documents/Onderzoek/ManifoldGPs/"
Random.seed!(61)

M = Hyperbolic(2)

# Starting point
a₀ = [0, 0.6] 
Y₀ = (1-dot(a₀,a₀))/2*[1 0 ; 0 1]
x₀, ν₀ = get_frame_vectors(a₀,Y₀,M)

# time Scale
T = 1.0
τ(s,T) = s*(2.0 - s/T)
tt_ = 0.0:0.001:T
tt = τ.(tt_,T)


# Ending point
xT = [x₀[1], -x₀[2], x₀[3]] 
νT = [-ν₀[1,1] ν₀[1,2] ; ν₀[2,1] -ν₀[2,2] ; ν₀[3,1] -ν₀[3,2]]
aT, YT = get_frame_parameterized(xT, νT, M)
obs = observation(T, (xT, νT))


inner_product_local(M,a₀,X,Y) = local_metric(M,a₀)*dot(X,Y)
minkowski_metric(M,X,Y) = -X[end]*Y[end]+dot(X[1:end-1],Y[1:end-1])

function is_local_orthonormal(M,a,Y)
    out1 = isapprox(inner_product_local(M,a₀,Y[:,1], Y[:,1]), 1.0)
    out2 = isapprox(inner_product_local(M,a₀,Y[:,2], Y[:,2]), 1.0)
    out3 = isapprox(inner_product_local(M,a₀,Y[:,1], Y[:,2]), 0.0)
    return all((out1,out2,out3))
end

function is_minkowski_orthonormal(M,x,ν)
    out1 = isapprox(minkowski_metric(M,ν[:,1], ν[:,1]), 1.0)
    out2 = isapprox(minkowski_metric(M,ν[:,2], ν[:,2]), 1.0)
    out3 = isapprox(minkowski_metric(M,ν[:,1], ν[:,2]), 0.0)
    return all((out1,out2,out3))
end

@assert is_local_orthonormal(M,a₀,Y₀) "Initial frame not locally orthonormal"
@assert is_minkowski_orthonormal(M,x₀,ν₀) "Initial frame not minkowski orthonormal"

# sample random numbers for MC-approximation of the guiding term
Random.seed!(6)
Z = randn(5000)
Zpos = Z[Z.>-2]

# set vector field
V(θ) = (a) -> θ*a

# generate one path of the guided process
drift(M,t,a) = V(-20)(a) + ∇logg(M,t,a,obs, Zpos)
W = sample(tt, Wiener{SVector{2,Float64}}())
U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
aa = map(x -> convert(PoincareBallPoint, x).value, xx) # this contains the points we plot

# simulate 4 paths for vector field V(a)=-20a and write to csv
someguidedpaths = Matrix{Float64}(undef, 0, 4)
for i in 1:4
    W = sample(tt, Wiener{SVector{2,Float64}}())
    U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
    xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
    aa = map(x -> convert(PoincareBallPoint, x).value, xx) # this contains the points we plot
    aamat = hcat(tt, getindex.(aa,1), getindex.(aa,2), fill(i, length(tt)))
    someguidedpaths = vcat(someguidedpaths, aamat)    
end
writedlm(outdir*"someguidedpaths_minus20.csv", someguidedpaths , ',')

# simulate 4 paths for vector field V(a)=a and write to csv
drift(M,t,a) = V(1)(a) + ∇logg(M,t,a,obs, Zpos)
someguidedpaths = Matrix{Float64}(undef, 0, 4)
for i in 1:4
    W = sample(tt, Wiener{SVector{2,Float64}}())
    U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
    xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
    aa = map(x -> convert(PoincareBallPoint, x).value, xx) # this contains the points we plot
    aamat = hcat(tt, getindex.(aa,1), getindex.(aa,2), fill(i, length(tt)))
    someguidedpaths = vcat(someguidedpaths, aamat)    
end
writedlm(outdir*"someguidedpaths_1.csv", someguidedpaths , ',')

# simulate 4 paths for vector field V(x) = [5.0*(1.0-x[1]^2-x[2]^2)^2, 0.0] and write to csv
V(θ)(a) = [θ*(1.0-a[1]^2-a[2]^2)^2, 0.0]
drift(M,t,a) = V(5)(a) + ∇logg(M,t,a,obs, Zpos)
someguidedpaths = Matrix{Float64}(undef, 0, 4)
for i in 1:4
    W = sample(tt, Wiener{SVector{2,Float64}}())
    U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
    xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
    aa = map(x -> convert(PoincareBallPoint, x).value, xx) # this contains the points we plot
    aamat = hcat(tt, getindex.(aa,1), getindex.(aa,2), fill(i, length(tt)))
    someguidedpaths = vcat(someguidedpaths, aamat)    
end
writedlm(outdir*"someguidedpaths_right.csv", someguidedpaths , ',')

############################################################# 
# pCN for V3 = [5.0*(1.0-x[1]^2-x[2]^2)^2, 0.0]
V(θ)(a) = [θ*(1.0-a[1]^2-a[2]^2)^2, 0.0]
drift(M,t,a) = V(5)(a) + ∇logg(M,t,a,obs, Zpos)

W = sample(tt, Wiener{SVector{2,Float64}}())
U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
aa = map(x -> convert(PoincareBallPoint, x).value, xx) # this contains the points we plot


# Crank nicolson scheme
nr_simulations = 500
λ = 0.4

X = deepcopy(U)
xx = map(x -> x[1], X.yy)
samplepaths = [xx]
acc = [true]
Xᵒ = deepcopy(X)
ll = loglikelihood(Xᵒ, obs, M, Zpos)
llvals = [ll]

prog = Progress(nr_simulations)
for k in 1:nr_simulations
    Wᵒ = crank_nicolson(λ, W)
    StochasticDevelopment!(heun(), Xᵒ, Wᵒ, drift, (x₀, ν₀), M)
    llᵒ = loglikelihood(Xᵒ, obs, M, Zpos)
    llᵒ = isnan(llᵒ) ? -1e10 : llᵒ
    if log(rand()) <= llᵒ - ll
        push!(acc,true)
        W = Wᵒ
        X = Xᵒ
        ll = llᵒ
    else
        push!(acc, false)
    end
    push!(samplepaths, map(x -> x[1], X.yy))
    push!(llvals, ll)
    next!(prog)
end
println(mean(acc))

# save all paths
mcmcpaths = Matrix{Float64}(undef, 0, 4)
for i in eachindex(samplepaths)
    aa = map(x -> convert(PoincareBallPoint, x).value, samplepaths[i])
    aamat = hcat(tt, getindex.(aa,1), getindex.(aa,2), fill(i, length(tt)))
    mcmcpaths = vcat(mcmcpaths, aamat)    
end
writedlm(outdir*"mcmcpaths.csv", mcmcpaths, ',')


##################### 
# one more experiment where we sample the guided process
# with V3 = [5.0*(1.0-x[1]^2-x[2]^2)^2, 0.0] have a path start on the right
# and end at the left

# Starting point
a₀ = [0.2, 0.0] 
Y₀ = (1-dot(a₀,a₀))/2*[1 0 ; 0 1]
x₀, ν₀ = get_frame_vectors(a₀,Y₀,M)

# Ending point
xT = [-0.8, x₀[2], x₀[3]] 
νT = [-ν₀[1,1] ν₀[1,2] ; ν₀[2,1] -ν₀[2,2] ; ν₀[3,1] -ν₀[3,2]]

aT, YT = get_frame_parameterized(xT, νT, M)
obs = observation(T, (xT, νT))


someguidedpaths = Matrix{Float64}(undef, 0, 4)
for i in 1:4
    W = sample(tt, Wiener{SVector{2,Float64}}())
    U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
    xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
    aa = map(x -> convert(PoincareBallPoint, x).value, xx) # this contains the points we plot
    aamat = hcat(tt, getindex.(aa,1), getindex.(aa,2), fill(i, length(tt)))
    someguidedpaths = vcat(someguidedpaths, aamat)    
end
writedlm(outdir*"someguidedpaths_right_difficult.csv", someguidedpaths , ',')

