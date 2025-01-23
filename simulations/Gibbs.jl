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
Random.seed!(6)
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
TimeChange(T) = (x) ->  x * (1.7+(1-1.7)*x/T)
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
        return V(θ, Φ)(a) .+ ∇logg(M,B,t,a, obs[k])
    end
    return out
end

function drift_multiple_obs(θ, trunc, obs)
    function out(M, B, t, a)
        k = getk(map(x -> x.t, obs), t)
        return V(θ, Φ)(a) .+ ∇logg(M,B,t,a,trunc, obs[k])
    end
    return out
end

θ = rand(MvNormal(zeros(2), inv(Γ₀)))
    W = sample(tt, Wiener{SVector{2, Float64}}())
    X = StochasticDevelopment(heun(), W, drift_multiple_obs(θ, obs), (x₀, ν), M, A)
    ll = loglikelihood(X, obs, θ, Φ, M, A)
while isnan(ll)
    θ = rand(MvNormal(zeros(2), inv(Γ₀)))
    W = sample(tt, Wiener{SVector{2, Float64}}())
    X = StochasticDevelopment(heun(), W, drift_multiple_obs(θ, obs), (x₀, ν), M, A)
    ll = loglikelihood(X, obs, θ, Φ, M, A)
end
# Gibbs sampler
function gibbs(λ , nr_iterations, Γ₀, obs::Array{observation, 1},θ,W,X)
    # Prior draw of θ
    # θ1 = rand(MvNormal(zeros(2), inv(Γ₀)))
    # θ20 = rand(MvNormal(zeros(2), inv(Γ₀)))
    # θ = rand(MvNormal(zeros(2), inv(Γ₀)))
    # W1 = sample(tt, Wiener{SVector{2, Float64}}())
    # W20 = deepcopy(W1)
    # W = sample(tt, Wiener{SVector{2, Float64}}())
    # X1 = StochasticDevelopment(heun(), W, drift_multiple_obs(θ1, 1, obs), (x₀, ν), M, A)
    # X20 = StochasticDevelopment(heun(), W, drift_multiple_obs(θ20, 20, obs), (x₀, ν), M, A)
    # X1ᵒ = deepcopy(X1)
    # X20ᵒ = deepcopy(X20)
    # X = StochasticDevelopment(heun(), W, drift_multiple_obs(θ, 10, obs), (x₀, ν), M, A)
    # Xᵒ = copy(X)
    # ll1 = loglikelihood(X1, obs, θ1, Φ, M, A)
    # ll20 = loglikelihood(X20, obs, θ20, Φ, M, A)
    # ll = loglikelihood(X, obs, θ, Φ, M, A)
    prog = Progress(nr_iterations)
    # acc_W1 = 0
    # acc_W20 = 0
    acc_W = 0
    # θ1_array = [θ1]
    # θ20_array = [θ20]
    θ_array = [θ]
    for j in 1:nr_iterations
        # Update (X,W) given θ
        # StochasticAntiDevelopment!(W, X1, drift_multiple_obs(θ1, 1, obs), M, A)
        # StochasticAntiDevelopment!(W, X20, drift_multiple_obs(θ20, 20, obs), M, A)
        StochasticAntiDevelopment!(W, X, drift_multiple_obs(θ, 10,obs), M, A)
        # ll1 = loglikelihood(X1, obs, θ1, Φ, M, A)
        # ll20 = loglikelihood(X20, obs, θ20, Φ, M, A)
        ll = loglikelihood(X, obs, θ, Φ, M, A)
        Wᵒ = crank_nicolson(λ, W)
        # StochasticDevelopment!(heun(), X1ᵒ,  Wᵒ, drift_multiple_obs(θ1, 1, obs), (x₀, ν), M, A)
        # StochasticDevelopment!(heun(), X20ᵒ,  Wᵒ, drift_multiple_obs(θ20, 20, obs), (x₀, ν), M, A)
        Xᵒ = StochasticDevelopment(heun(), Wᵒ, drift_multiple_obs(θ,10,obs), (x₀, ν), M, A)
        # ll1ᵒ = loglikelihood(X1ᵒ, obs, θ1, Φ, M, A)
        # ll20ᵒ = loglikelihood(X20ᵒ, obs, θ20, Φ, M, A)
        llᵒ = loglikelihood(Xᵒ, obs, θ, Φ, M, A)
        llᵒ = isnan(llᵒ) ? 1e-10 : llᵒ
        # if log(rand()) <= ll1ᵒ - ll1 
        #     ll1 = ll1ᵒ
        #     W1 = Wᵒ
        #     X1 = X1ᵒ
        #     acc_W1 += 1
        # end
        # if log(rand()) <= ll20ᵒ - ll20
        #     ll20 = ll20ᵒ
        #     W20 = Wᵒ
        #     X20 = X20ᵒ
        #     acc_W20 += 1
        # end
        if log(rand()) <= llᵒ - ll
            # println("Accepted: llᵒ = $llᵒ , ll = $ll ")
            ll = llᵒ
            W = Wᵒ
            X = Xᵒ
            acc_W += 1
        end
        # _μ1, _Γ1 = μΓ(M, A, X1, Φ)
        # _μ20, _Γ20 = μΓ(M, A, X20, Φ)
        # dump(X)
        _μ, _Γ = μΓ(M, A, X, Φ)
        _Γ += Γ₀
        # θ1 = rand(MvNormal( _Γ1 \ _μ1 , inv(_Γ1)))
        # θ20 = rand(MvNormal( _Γ20 \ _μ20 , inv(_Γ20)))   
        θ = rand(MvNormal( _Γ \ _μ , inv(_Γ)))
        # push!(θ1_array, θ1)
        # push!(θ20_array, θ20)
        push!(θ_array, θ)
        next!(prog)
    end
    # println("Bridge K = 1 acceptance rate: $( 100*acc_W1/(nr_iterations) )%")
    # println("Bridge K = 20 acceptance rate: $( 100*acc_W20/(nr_iterations) )%")
    println("Bridge K = $truncation acceptance rate: $( 100*acc_W/(nr_iterations) )%")
    # return θ1_array, X1, θ20_array, X20 # concat(bridges)
    return θ_array, X
end

nr_iterations = 1000
Γ₀ = diagm([0.05, 0.05])
θ_array, X  = gibbs(0.97, nr_iterations, Γ₀, obs,θ,W,X)

Wᵒ = crank_nicolson(0.9, W)
Xᵒ = StochasticDevelopment(heun(), Wᵒ, drift_multiple_obs(θ,  obs), (x₀, ν), M, A)


using Plots
plt = hist(map(x -> x[1], θ1_array))
plt.plot[1][]
f = Figure()
stephist(map(x -> x[1], θ1_array))
stephist!(map(x -> x[1], θ20_array))
Plots.scalefontsizes(2.5)
h1 = fit(Histogram, map(x -> x[1], θ1_array), -4.0:0.2:1.0 )
h20 = fit(Histogram, map(x -> x[1], θ20_array),  -4.0:0.2:1.0 )
hdiff = fit(Histogram, map(x -> x[1], θ1_array), -4.0:0.2:1.0)
hdiff.weights = hdiff.weights .- h20.weights
vals = fit(Histogram, map(x -> x[1], θ1_array), nbins = 15).edges
height = fit(Histogram, map(x -> x[1], θ1_array), nbins = 15).weights


fig = let
    h1 = fit(Histogram, map(x -> x[1], θ1_array), -4.0:0.2:1.0 )
    h20 = fit(Histogram, map(x -> x[1], θ20_array),  -4.0:0.2:1.0 )
    hdiff = fit(Histogram, map(x -> x[1], θ1_array), -4.0:0.2:1.0)
    hdiff.weights = hdiff.weights .- h20.weights
    p1 = Plots.plot(h1, linetype = :steppost, 
                size = (1800,1600), label = L"$K=1$",  margin = 10Plots.mm,
                legend = :topleft, xlabel = L"$\theta_1$", ylabel = "Frequency", linewidth = 5.5, dpi = 300)
        Plots.plot!(p1, h20,  label = L"$K=20$", linetype = :steppost,linestyle = :dash, linewidth = 5.5, dpi = 300)
        Plots.plot!(p1, hdiff, label = "difference" , linetype = :steppost, fillrange = 0.0, fillalpha = 0.7)
    h1 = fit(Histogram, map(x -> x[2], θ1_array), 1.0:0.1:3.50 )
    h20 = fit(Histogram, map(x -> x[2], θ20_array),  1.0:0.1:3.5 )
    hdiff = fit(Histogram, map(x -> x[2], θ1_array), 1.0:0.1:3.5)
    hdiff.weights = hdiff.weights .- h20.weights
    p2 = Plots.plot(h1, linetype = :steppost, 
        size = (1800,1600), label = L"$K=1$",  margin = 10Plots.mm,
        legend = :topleft, xlabel = L"$\theta_2$", ylabel = "Frequency", linewidth = 5.5, dpi = 300)
    Plots.plot!(p2, h20, linetype = :steppost, label = L"$K=20$", linestyle = :dash, linewidth = 5.5, dpi = 300)
    Plots.plot!(p2, hdiff, label = "difference" , linetype = :steppost, fillrange = 0.0, fillalpha = 0.7)
    fig = Plots.plot(p1, p2, layout = (2,1), size = (1800,1600))
    fig
end
savefig(fig, outdir*"effect_of_truncation.png")

CairoMakie.activate!()
fig = let 
    fig = stephist(map(x -> x[1], θ1_array), 
            normalize = :pdf, 
            size = (1800,1600), 
            label = L"$K=1$",  
            margin = 10Plots.mm,
            legend = :topleft, 
            xlabel = L"$\theta_1$", 
            ylabel = "Density", 
            linewidth = 5.5, 
            dpi = 300)
    stephist!(fig, map(x -> x[1], θ20_array), 
            normalize = :pdf, 
            size = (1800,1600), 
            label = L"$K=1$",  
            margin = 10Plots.mm,
            legend = :topleft, 
            xlabel = L"$\theta_1$", 
            ylabel = "Density", 
            linewidth = 5.5, 
            linestyle = :dash,
            dpi = 300)
    fig
end

CairoMakie.activate!()
fig = let
    burnin = 100
    fig = Figure( resolution=(2500, 2500), size = (1200,1200), fontsize=35)
    ax1 = Axis(fig[1, 1] , xlabel = L"$\theta_1$", ylabel = "Density")
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
Makie.save(outdir*"gibbs_samples_4-4.png", fig)

using DelimitedFiles
writedlm(outdir*"gibbs_samples_normal_prior.csv",θ_array,',')
θ_array = readdlm(outdir*"/manifold_bridge.jl/gibbs_samples_normal_prior.csv",',')
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

