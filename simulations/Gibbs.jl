include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")
using ProgressMeter
using Random
using Distributions
outdir = "/Users/marc/Documents/Onderzoek/ManifoldGPs/"

function linear_combination(θ, ϕ::Array{T, 1}) where {T<:Function}
    K = typeof(θ) <: Real ? 1 : length(θ)
    return x -> K == 1 ? θ*ϕ[1](x) : sum([θ[k]*ϕ[k](x) for k in 1:K])
end
ϕ1(a) = [1., 0.]
ϕ2(a) = [0., 1.]
Φ = [ϕ1, ϕ2]
V(θ, Φ) = linear_combination(θ, Φ)


# Simulate data
Random.seed!(61)
M = Manifolds.EmbeddedTorus(3,2)
A = Manifolds.DefaultTorusAtlas()

# Initial point
x₀ = [3.0, 0.0, 2.0]
# Place in an induced basis 
i = Manifolds.get_chart_index(M, A, x₀)
a₀ = Manifolds.get_parameters(M, A, i, x₀)
B = induced_basis(M,A,i)
# Basis for tangent space to M at x₀
N = Array{Float64}(Manifolds.normal_vector(M,x₀))
ν = nullspace(N')


θ₀ = [4.0, -4.0]
# TimeChange(T) = (x) ->  x * (2-x/T)
T = 1.0
dt = 0.001
tt =  0.0:dt:T # map((x) ->  x * (2-x/T) , 0.0:dt:T)
W = sample(tt, Wiener{SVector{2, Float64}}())
X₀ = StochasticDevelopment(heun(), W , (M,B,t,a) -> V(θ₀, Φ)(a) , (x₀,ν),  M, A)
n = 40
ind = pushfirst!(Int64.(collect((length(tt)-1)/n:(length(tt)-1)/n:length(tt))), 1)
ind[end] = length(tt)
times = X₀.tt[ind]
obs = [observation(X₀.tt[i], X₀.yy[i]) for i in ind]


function getk(times::Array{T,1}, t::T) where {T<:Real}
    k = searchsortedfirst(times, t)
    if times[k] == t 
        k += 1
    end
    return k
end #  t in [times[k-1], times[k])

function drift_multiple_obs(θ, obs)
    function out(M, B, t, a)
        k = getk(map(x -> x.t, obs), t)
        return V(θ, Φ)(a) .+ ∇logg(M,B,t,a,obs[k])
    end
    return out
end


# Gibbs sampler
function gibbs(λ , nr_iterations, Γ₀, obs::Array{observation, 1})
    # Prior draw of θ
    θ = rand(MvNormal(zeros(2), inv(Γ₀)))
    W = sample(tt, Wiener{SVector{2, Float64}}())
    X = StochasticDevelopment(heun(), W, drift_multiple_obs(θ, obs), (x₀, ν), M, A)
    Xᵒ = deepcopy(X)
    ll = loglikelihood(X, obs, θ, Φ, M, A)
    prog = Progress(nr_iterations)
    acc_W = 0
    θ_array = [θ]
    for j in 1:nr_iterations
        # Update (X,W) given θ
        StochasticAntiDevelopment!(W, X, drift_multiple_obs(θ, obs), M, A)
        ll = loglikelihood(X, obs, θ, Φ, M, A)
        Wᵒ = crank_nicolson(λ, W)
        StochasticDevelopment!(heun(), Xᵒ,  Wᵒ, drift_multiple_obs(θ, obs), (x₀, ν), M, A)
        llᵒ = loglikelihood(Xᵒ, obs, θ, Φ, M, A)
        if log(rand()) <= llᵒ - ll 
            ll = llᵒ
            W = Wᵒ
            X = Xᵒ
            acc_W += 1
        end
        _μ, _Γ = μΓ(M, A, X, Φ)
        θ = rand(MvNormal( _Γ \ _μ , inv(_Γ)))
        push!(θ_array, θ)
        next!(prog)
    end
    println("Bridge acceptance rate: $( 100*acc_W/(nr_iterations) )%")
    return θ_array, X # concat(bridges)
end

nr_iterations = 1000
Γ₀ = diagm([0.005, 0.005])
θ_array, X = gibbs(0.9, nr_iterations, Γ₀, obs)


CairoMakie.activate!()
fig = let
    burnin = 100
    fig = Figure( resolution=(2500, 2500), size = (1200,1200), fontsize=35)
    ax1 = Axis(fig[1, 1] , xlabel = L"$θ_1$", ylabel = "Density")
    Makie.hist!(ax1, map(x -> x[1], θ_array[burnin:end]) ; bins = 15 ,  color = palette(:default)[1], normalization = :pdf, flip = false)
    Makie.vlines!(ax1, [θ₀[1]] ; linewidth = 5.0, color = palette(:default)[2] , label = "True value")
    axislegend(ax1; 
            labelsize = 35, 
            framewidth = 1.0, 
            labelfont = :bold, 
            orientation = :vertical,
            patchlabelgap = 18,
            patchsize = (50.0,50.0),
            margin = (20.0,20.0,20.0,20.0))
    ax2 = Axis(fig[2,1], xlabel = L"$θ_2$", ylabel = "Density")
    Makie.hist!(ax2, map(x -> x[2], θ_array[burnin:end]) ; bins = 15, color = palette(:default)[1], normalization = :pdf, flip = false)
    Makie.vlines!(ax2, [θ₀[2]] ; linewidth = 5.0, color = palette(:default)[2], label = "True value")
    axislegend(ax2; 
            labelsize = 35, 
            framewidth = 1.0, 
            labelfont = :bold, 
            orientation = :vertical,
            patchlabelgap = 18,
            patchsize = (50.0,50.0),
            margin = (20.0,20.0,20.0,20.0))
    ax3 = Axis(fig[1, 2],xlabel = "Iteration", ylabel = L"$\theta_1$")
    Makie.lines!(ax3, collect(1:length(θ_array)) , map(x -> x[1], θ_array) ; linewidth = 3.0, color = palette(:default)[1], label = "Trace of sampler")
    Makie.hlines!(ax3, [θ₀[1]] ; color =  palette(:default)[2], label = " True value", linewidth = 3.0)
    axislegend(ax3; 
            labelsize = 35, 
            framewidth = 1.0, 
            labelfont = :bold, 
            orientation = :vertical,
            patchlabelgap = 18,
            patchsize = (50.0,50.0),
            margin = (20.0,20.0,20.0,20.0))
    ax4 = Axis(fig[2, 2],xlabel = "Iteration", ylabel = L"$\theta_2$")
    Makie.lines!(ax4, collect(1:length(θ_array)) , map(x -> x[2], θ_array) ; linewidth = 3.0, color = palette(:default)[1], label = "Trace of sampler")
    Makie.hlines!(ax4, [θ₀[2]] ; color =  palette(:default)[2], label = " True value", linewidth = 3.0)
    axislegend(ax4; labelsize = 35, 
            framewidth = 1.0, 
            labelfont = :bold, 
            orientation = :vertical,
            patchlabelgap = 18,
            patchsize = (50.0,50.0),
            margin = (20.0,20.0,20.0,20.0))
    fig
end
Makie.save(outdir*"gibbs_samples.png", fig)

using DelimitedFiles
writedlm(outdir*"gibbs_samples_normal_prior.csv",θ_array,',')

GLMakie.activate!()
fig = let
    ax, fig = torus_figure(M)
    lines!(ax, map(x -> x[1], map(x -> x[1], X.yy)), 
                    map(x -> x[2], map(x -> x[1], X.yy)) , 
                    map(x -> x[3], map(x -> x[1], X.yy)) ; 
                    linewidth = 3.0, color = palette(:default)[1], label = L"$X")             
    Makie.scatter!(ax, obs[1].u[1][1], obs[1].u[1][2], obs[1].u[1][3], color = palette(:default)[2], markersize = 25, label = L" $x_i$")
    for i in 2:n+1
        Makie.scatter!(ax, obs[i].u[1][1],obs[i].u[1][2],obs[i].u[1][3], color = palette(:default)[2], markersize = 25)
    end
    axislegend(ax; 
                    labelsize = 50, 
                    framewidth = 1.0, 
                    orientation = :vertical,
                    patchlabelgap = 18,
                    patchsize = (50.0,50.0),
                    margin = (320.0,320.0,320.0,320.0))
    fig
end