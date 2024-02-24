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
# TimeChange(T) = (x) ->  x * (2-x/T) # Optional
T = 1.0
dt = 0.001
tt =  0.0:dt:T # map((x) ->  x * (2-x/T) , 0.0:dt:T)
W = sample(tt, Wiener{SVector{2, Float64}}())
X₀ = StochasticDevelopment(heun(), W, (M,B,t,a) -> V(θ₀, Φ)(a), (x₀,ν), M, A)
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
# pseudo-marginal MCMC
function pseudo_marginal_MH(nr_iterations, nr_samples_for_estimator, σ, Γ₀, obs::Array{observation, 1})
    n = length(obs)-1
    θ = rand(MvNormal(zeros(2), inv(Γ₀)))
    bridges = separate(X₀, ind)
    bridgesᵒ = separate(X₀, ind)
    θ_array = [θ]
    acc = 0
    prog = Progress(nr_iterations)
    for j in 1:nr_iterations
        θᵒ = rand(MvNormal(θ, diagm(σ)))
        ll = zeros(n) ; llᵒ = zeros(n)
        for k in 1:nr_samples_for_estimator
            W = sample(tt, Wiener{SVector{2, Float64}}())
            Wᵒ = sample(tt, Wiener{SVector{2, Float64}}())
            BMs = separate(W,ind)
            BMsᵒ = separate(Wᵒ, ind)
            for i in 1:n
                drift(M,B,t,a) = V(θ, Φ)(a) .+ ∇logg( M,B,t,a,obs[i+1] )
                driftᵒ(M,B,t,a) = V(θᵒ, Φ)(a) .+ ∇logg( M,B,t,a,obs[i+1] )
                bridges[i] = StochasticDevelopment(heun(), BMs[i], drift, obs[i].u, M, A)
                bridgesᵒ[i] = StochasticDevelopment(heun(), BMsᵒ[i], driftᵒ, obs[i].u, M, A)
                ll[i] += exp(loglikelihood(bridges[i], obs[i+1], θ,Φ, M,A))/nr_samples_for_estimator
                llᵒ[i] += exp(loglikelihood(bridgesᵒ[i], obs[i+1], θᵒ,Φ, M,A))/nr_samples_for_estimator
            end        
        end
        logacc = sum(log.(llᵒ)) - sum(log.(ll)) #- 0.5*dot(θᵒ, Γ₀\θᵒ) + 0.5*dot(θ, Γ₀\θ) # log(llᵒ) - log(ll)  #) #sum(llᵒ)- sum(ll)
        if log(rand()) <= logacc
            θ = θᵒ
            acc += 1
        end
        next!(prog)
        push!(θ_array, θ)
    end
    println("Acceptence rate: $(round(100*acc/nr_iterations ; digits = 3))%")
    return θ_array
end

nr_iterations = 1000
nr_samples_for_estimator = 10
Γ₀ = diagm([0.005, 0.005])
σ = [1.0, 1.0]
θ_array = pseudo_marginal_MH(nr_iterations, nr_samples_for_estimator, σ, Γ₀, obs)

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

Makie.save(outdir*"pseudo_marginal_MH_samples.png", fig)

using DelimitedFiles
writedlm(outdir*"pseudo_marginal_MH_samples_uniform_prior.csv",θ_array,',')


GLMakie.activate!()
fig = let
    ax, fig = torus_figure(M)
    lines!(ax, map(x -> x[1], map(x -> x[1], X.yy)), 
                    map(x -> x[2], map(x -> x[1], X.yy)) , 
                    map(x -> x[3], map(x -> x[1], X.yy)) ; 
                    linewidth = 4.0, color = palette(:default)[1], label = L"$X")             
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

