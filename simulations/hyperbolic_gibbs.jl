using Pkg
Pkg.activate()
using ProgressMeter
using Random
using Distributions
using Plots
include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")
outdir = "/Users/marc/Documents/Onderzoek/ManifoldGPs/"

Random.seed!(6)
M = Hyperbolic(2)
V(θ) = (a) -> [θ*(1-dot(a,a))^2, 0.0]
#Initial point
a₀ = [-0.6, 0.0] 
Y₀ = (1-dot(a₀,a₀))/2*[1 0 ; 0 1]
x₀, ν₀ = get_frame_vectors(a₀,Y₀,M)


# time Scale
T = 1.0
tt = 0.0:0.001:T

θ₀ = 5
W = sample(tt, Wiener{SVector{2, Float64}}())
drift(M,t,a) = V(θ₀)(a)
X = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
n = 100
ind = pushfirst!(Int64.(collect((length(tt)-1)/n:(length(tt)-1)/n:length(tt))), 1)
ind[end] = length(tt)
times = X.tt[ind]
obs = [observation(X.tt[i], X.yy[i]) for i in ind]


function getk(times::Array{T,1}, t::T) where {T<:Real}
    k = searchsortedfirst(times, t)
    if times[k] == t 
        k += 1
    end
    return k
end #  t in [times[k-1], times[k])

function drift_multiple_obs(θ, obs, Zpos)
    function out(M, t, a)
        k = getk(map(x -> x.t, obs), t)
        return V(θ)(a) .+ ∇logg(M,t,a, obs[k], Zpos)
    end
    return out
end

# Gibbs sampler
function gibbs(λ, nr_iterations, Γ₀, obs::Array{observation, 1}, θ, W, X, Zpos)
    prog = Progress(nr_iterations)
    acc_W = 0
    θ_array = [θ]
    for j in 1:nr_iterations
        StochasticAntiDevelopment!(W, X, drift_multiple_obs(θ, obs, Zpos), M)
        ll = loglikelihood(X, obs, V(θ), M, Zpos)
        Wᵒ = crank_nicolson(λ, W)
        Xᵒ = StochasticDevelopment(heun(), Wᵒ, drift_multiple_obs(θ, obs, Zpos), (x₀, ν₀), M)
        llᵒ = loglikelihood(Xᵒ, obs,V(θ), M, Zpos)
        llᵒ = isnan(llᵒ) ? 1e-10 : llᵒ
        if log(rand()) <= llᵒ - ll
            # println("Accepted: llᵒ = $llᵒ , ll = $ll ")
            ll = llᵒ
            W = Wᵒ
            X = Xᵒ
            acc_W += 1
        end
        _μ, _Γ = μΓ(M, X, V)
        _Γ += Γ₀ 
        θ = _Γ\_μ + randn()/sqrt(_Γ)
        push!(θ_array, θ)
        next!(prog)
    end
    return θ_array, X, acc_W
end

nr_iterations = 1000
Γ₀ = 0.05
θ = randn()/sqrt(Γ₀)
Z = randn(500)
Zpos = Z[Z.>-2]
θ_array, X, acc = gibbs(0.9,nr_iterations,Γ₀,obs,θ,W,X,Zpos)

CairoMakie.activate!()
fig = let
    burnin = 100
    fig = Figure( resolution=(2500, 2500), size = (1200,1200), fontsize=55)
    ax1 = Axis(fig[1, 1] , xlabel = L"$\theta$", ylabel = "Density")
    Makie.hist!(ax1, map(x -> x[1], θ_array[burnin:end]) ; 
                                color= (Makie.wong_colors()[1],0.4),
                                bins = 20, 
                                strokewidth = 2,
                                strokecolor = :black, 
                                normalization = :pdf)
    Makie.vlines!(ax1, [θ₀] ; linewidth = 7.0, color = Makie.wong_colors()[2] , label = "True value")
    axislegend(ax1; 
            labelsize = 55, 
            framewidth = 1.0, 
            labelfont = :bold, 
            orientation = :vertical,
            patchlabelgap = 18,
            patchsize = (50.0,50.0),
            margin = (20.0,20.0,20.0,20.0))
    ax1.xlabelsize = 85
    ax1.ylabelsize = 55
    ax2 = Axis(fig[2, 1], xlabel = "Iteration", ylabel = L"$\theta$")
    Makie.lines!(ax2, collect((1):length(θ_array)) ,θ_array[1:end] ; 
                    color = (Makie.wong_colors()[1], 0.6), 
                    linewidth = 3.0,  label = "Trace of sampler")
    Makie.hlines!(ax2, [θ₀] ; color = Makie.wong_colors()[2], label = " True value", linewidth = 7.0)
    axislegend(ax2; 
            labelsize = 55, 
            framewidth = 1.0, 
            labelfont = :bold, 
            orientation = :vertical,
            patchlabelgap = 18,
            patchsize = (50.0,50.0),
            margin = (20.0,20.0,20.0,20.0))
    ax2.xlabelsize = 55
    ax2.ylabelsize = 85
    fig
end
Makie.save(outdir*"gibbs_samples_hyperbolic_right_vf.png", fig)


CairoMakie.activate!()
fig = let
    xx = map(u -> u[1], X.yy) ; νν = map(u -> u[2], X.yy)
    aa = map(x -> convert(PoincareBallPoint, x).value, xx)
    fig = Figure()
    ax = Axis(fig[1, 1], yautolimitmargin = (0.1, 0.1), xautolimitmargin = (0.1, 0.1))
    arc!(Point2f(0), 1, -π, π, color = :red)
    Makie.lines!(map(x -> x[1], aa[1:end-3]), map(x -> x[2], aa[1:end-3]), color = color = palette(:default)[1])
    for o in obs
        a = convert(PoincareBallPoint, HyperboloidPoint(o.u[1])).value
        Makie.scatter!(ax, a[1],a[2],color = palette(:default)[2], markersize =8)
    end
    fig
end
Makie.save(outdir*"gibbs_samples_final_path_center_vf.png", fig)

using DelimitedFiles
writedlm(outdir*"gibbs_samples_hyperbolic_center_vf.csv",θ_array,',')
