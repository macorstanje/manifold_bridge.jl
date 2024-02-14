include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")
GLMakie.activate!()
using Random
using Distributions


"""
    General forward simulation on the Torus
"""

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
# Standard Brownian motion
tt = 0.0:0.001:10.0
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
drift(M,B,t,a) = zero(a)
X = StochasticDevelopment(heun(), W, drift, (x₀, ν) , M, A)
# X = HorizontalDevelopment(heun(), γ, (p,ν), M, A)
xx = map(x -> x[1], X.yy)
ax1, fig1 = torus_figure(M)
lines!(ax1, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 2.0, color = :green)
fig1


# Horizontal development / geodesic on M
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
γ = deepcopy(W)
for i in eachindex(γ.tt) γ.yy[i] = 1.0.*[γ.tt[i], 2*γ.tt[i]] end
X = HorizontalDevelopment(heun(), γ, (x₀,ν), M, A)
# Plot geodesic with some frames illustrated
xx = map(x -> x[1], X.yy)
νν = map(x -> x[2], X.yy)
ind = 1:300:length(tt)

pts_array = [Point3f(xx[i]) for i in ind]
vec1_array = [Point3f(νν[i][:,1]) for i in ind]
vec2_array = [Point3f(νν[i][:,2]) for i in ind]

ax, fig = torus_figure(M)
lines!(ax, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 15.0, color = :green)
arrows!(ax, pts_array, vec1_array, linewidth=0.04, color=:blue, transparency = false)
arrows!(ax, pts_array, vec2_array, linewidth=0.04, color=:blue, transparency = false)
fig
# Makie.save("horizontal_curve.png", fig)

# Plot initial frame
ax, fig = torus_figure(M)
Makie.scatter!(ax, x₀[1],x₀[2],x₀[3], color = :red, markersize = 45, label = "x₀")
arrows!(ax, [Point3f(xx[1])], [Point3f(2.0.*νν[1][:,1])], linewidth=0.04, color=:red, transparency = false)
arrows!(ax, [Point3f(xx[1])], [Point3f(2.0.*νν[1][:,2])], linewidth=0.04, color=:red, transparency = false)
fig
# Makie.save("frame.png", fig)



"""
    Brownian bridge simulation
"""
struct observation
    t::Real
    u::Tuple{Vector{Float64}, Matrix{Float64}}
end

# Heat kernel with 2K terms in the series expansion
K = 10
function κ(t, y , z, M::Manifolds.EmbeddedTorus, B::AbstractBasis)
    θ1 = y[1] + B.i[1] ; ϕ1 = y[2] + B.i[2]
    θ2 = z[1] + B.i[1] ; ϕ2 = z[2] + B.i[2]
    out = sum([ exp(-(θ1 - θ2 - 2*pi*k)^2/(4*t/M.r^2) - (ϕ1 - ϕ2 - 2*pi*l)^2/(4*t/M.R^2)) for k in -K:K, l in -K:K])
    return out/(4*π*t/(M.R*M.r))
end

# Guiding function in local coordinates. 
g_param(M::Manifolds.EmbeddedTorus, B::AbstractBasis, t, a, xT) = κ(T-t, a, get_parameters(M, B.A, B.i, xT), M, B)

# Riemannian metric
function metric(M,B::AbstractBasis, a)
    sinθ, cosθ = sincos(a[1] + B.i[1])
    return diagm([(M.R + M.r*cosθ)^2 , M.r^2]) 
end

# Riemannian cometric
function cometric(M, B::AbstractBasis, a)
   sinθ, cosθ = sincos(a[1] + B.i[1])
    return diagm([(M.R + M.r*cosθ)^(-2) , 1/M.r^2])    
end

# Riemannian gradient of log g, in local coordinates
function ∇logg(M::Manifolds.EmbeddedTorus, B::AbstractBasis, t, a, xT)
    _∇ = ForwardDiff.gradient(a -> log(g_param(M,B,t,a,xT)), a)
    g⁺ = cometric(M,B, a)
    return g⁺*_∇
end
a₀ = Manifolds.get_parameters(M, B.A,B.i,x₀)
# Simulation, conditioned to hit xT
T = 1.0
xT = [-3.0, 0.0,2.0]
check_point(M,xT)
tt = 0.0:0.001:T
drift(M,B,t,a) = ∇logg(M,B,t,a,xT)
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
X = StochasticDevelopment(heun(), W, drift, (x₀, ν), M, A)
xx = map(x -> x[1], X.yy)

# Multiple samplepaths
samplepaths = [xx]
for k in 1:10
    W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
    StochasticDevelopment!(heun(), X,  W, drift, (x₀, ν), M, A)
    push!(samplepaths, map(x -> x[1], X.yy))
end

# Plot samplepaths
ax, fig = torus_figure(M)
for k in 1:10
    lines!(ax, map(x -> x[1], samplepaths[k]), 
                map(x -> x[2], samplepaths[k]) , 
                map(x -> x[3], samplepaths[k]) ; 
                linewidth = 4.0, color = palette(:default)[k])
end
Makie.scatter!(ax, x₀[1],x₀[2],x₀[3], color = :red, markersize = 25, label = L" $x_0$")
Makie.scatter!(ax, xT[1],xT[2],xT[3], color = :blue, markersize = 25, label = L" $x_T$")
axislegend(ax; 
        labelsize = 50, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (320.0,320.0,320.0,320.0))
fig
# Makie.save("BBT2.png", fig)


"""
    Bridge process with a vector field
"""

# Definition of the vector field in terms of a parameter θ
function linear_combination(θ, ϕ::Array{T, 1}) where {T<:Function}
    K = typeof(θ) <: Real ? 1 : length(θ)
    return x -> K == 1 ? θ*ϕ[1](x) : sum([θ[k]*ϕ[k](x) for k in 1:K])
end
V(θ, ϕ) = linear_combination(θ, ϕ)
ϕ1(x) = SVector{3, Float64}(x[2], -x[1], 0.0)
ϕ2(x) = SVector{3, Float64}(0., 0.,  1.)
Φ = [ ϕ1, ϕ2 ]

ϕ1(a) = [1., 0.]
ϕ2(a) = [0., 1.]
Φ = [ϕ1, ϕ2]
V(θ, Φ) = linear_combination(θ, Φ)


# Illustration of the vector field
N = 20 ; θ₀ = [4.0,4.0]
θs, φs = LinRange(-π, π, N), LinRange(-π, π, N)
pts = [Point3f(Manifolds._torus_param(M, θ, φ)) for θ in θs, φ in φs]
pts_array = [reshape(pts, N^2, 1)[i,1] for i in 1:N^2]
vecs_array = [zeros(3) for i in 1:N^2]
vecs = [reshape([V(θ₀./10, Φ)([θ, φ]) for θ in θs, φ in φs], N^2, 1)[i,1] for i in 1:N^2]
for j in eachindex(vecs_array)
    B = induced_basis(M, A, Manifolds.get_chart_index(M, A, pts[j]))
    Manifolds.get_vector_induced_basis!(M, vecs_array[j], pts_array[j], vecs[j], B)
    # vecs_array[j] = Point3f(vecs_array[j])
end
ax1, fig1 = torus_figure(M)
arrows!(ax1, pts_array, Point3f.(vecs_array) ; linewidth=0.05, color=palette(:default)[1])
fig1
# Makie.save("vector_field.png", fig1)

# Bridge process, conditioned to hit xT
T = 1.0
tt = 0.0:0.001:T
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
drift(M,B,t,a) = V(θ₀, Φ)(a) .+ ∇logg(M,B,t,a,xT)
X = StochasticDevelopment(heun(), W, drift, (x₀, ν), M, A)

# multiple samplepaths
xx = map(x -> x[1], X.yy)
samplepaths = [xx]
for k in 1:10
    W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
    StochasticDevelopment!(heun(), X, W, drift, (x₀, ν), M, A)
    push!(samplepaths, map(x -> x[1], X.yy))
end


ax, fig = torus_figure(M)
for k in 1:10
    lines!(ax, map(x -> x[1], samplepaths[k]), 
                map(x -> x[2], samplepaths[k]) , 
                map(x -> x[3], samplepaths[k]) ; 
                linewidth = 4.0, color = palette(:default)[k])
end
Makie.scatter!(ax, x₀[1],x₀[2],x₀[3], color = :red, markersize = 25, label = L" $x_0$")
Makie.scatter!(ax, xT[1],xT[2],xT[3], color = :blue, markersize = 25, label = L" $x_T$")
axislegend(ax; 
        labelsize = 50, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (320.0,320.0,320.0,320.0))
fig
# Makie.save("BBVT2.png", fig)




"""
    Likelihood computation for full observations of the process
"""





"""
    Likelihood computation
"""
function llikelihood(X::SamplePath, xT::T, θ, Φ, M, A) where {T}
    i = Manifolds.get_chart_index(M, A, X.yy[1][1])
    B = induced_basis(M,A,i)
    a = Manifolds.get_parameters(M, B.A, B.i, X.yy[1][1])
    out = log(g_param(M, B, X.tt[1] , a, xT)) # log g(0,x₀)
    # out = 0.0
    for k in 1:length(X.tt)-1
        t, x = X.tt[k], X.yy[k][1]
        i = Manifolds.get_chart_index(M, A, x)
        B = induced_basis(M,A,i)
        a = Manifolds.get_parameters(M, A, i, x)

        _∇ = ForwardDiff.gradient(a -> log(g_param(M,B,t,a,xT)), a)
        out += dot( V(θ, Φ)(a) , _∇ )*( X.tt[k+1] - t )
    end
    return out
end

function llikelihood(X::SamplePath, obs::Array{observation, 1}, θ, Φ, M, A) where {T}
    ti , xi = map(x -> x.t, obs) , map(x -> x.u[1], obs)
    out = 0.0
    for j in 1:length(obs)-1
        i = Manifolds.get_chart_index(M,A,xi[j]) ; B = induced_basis(M,A,i)
        a = Manifolds.get_parameters(M, A, i, xi[j])
        out += log(g_param(M,B, ti[j], a, xi[j+1]))
        filtered_tt = filter(t -> ti[j] < t <= ti[j+1], X.tt)
        for k in 1:length(filtered_tt)-1
            _∇ = ForwardDiff.gradient(a -> log(g_param(M,B, filtered_tt[k] ,a, xi[j+1] )), a)
            out += dot( V(θ, Φ)(a) , _∇ )*( filtered_tt[k+1] - filtered_tt[k] )
        end
    end
    return out
end

# llikelihood(X, xT, θ, ϕ ,M, A)

W = sample(tt, Wiener{SVector{2, Float64}}())
X = StochasticDevelopment(heun(), W, drift,(x₀,ν), M, A)


function crank_nicolson(η, W)
    W₂ = sample(W.tt, Wiener{SVector{length(W.yy[1]) , Float64}}())
    return SamplePath(W.tt, η.*W.yy + sqrt(1-η^2).*W₂.yy)
end


Random.seed!(6)
T = 1.0
tt = 0.0:0.001:T
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
θ₀ = 4.0
xT = [-3.0, 0.0, 2.0] ; check_point(M, xT)
drift(M,B,t,a) = V(θ₀, Φ)(a) .+ ∇logg(M,B,t,a,xT)
X = StochasticDevelopment(heun(), W, drift, (x₀, ν), M, A)
Xᵒ = deepcopy(X)
ll = llikelihood(X, xT, θ₀, Φ, M, A)

# multiple samplepaths
nr_simulatios = 500
xx = map(x -> x[1], X.yy)
samplepaths = [xx]
acc = 0
prog = Progress(nr_simulatios)
for k in 1:nr_simulatios
    Wᵒ = crank_nicolson(0.75, W)
    StochasticDevelopment!(heun(), Xᵒ, Wᵒ, drift, (x₀, ν), M, A)
    llᵒ = llikelihood(Xᵒ, xT, θ₀, Φ, M, A)
    if log(rand()) <= llᵒ - ll
        acc += 1
        W = Wᵒ
        X = Xᵒ
        ll = llᵒ
        push!(samplepaths, map(x -> x[1], X.yy))
    end
    next!(prog)
end


ax, fig = torus_figure(M)
for k in 1:10
    lines!(ax, map(x -> x[1], samplepaths[end-k+1]), 
                map(x -> x[2], samplepaths[end-k+1]) , 
                map(x -> x[3], samplepaths[end-k+1]) ; 
                linewidth = 4.0, color = palette(:default)[k])
end
Makie.scatter!(ax, x₀[1],x₀[2],x₀[3], color = :red, markersize = 25, label = L" $x_0$")
Makie.scatter!(ax, xT[1],xT[2],xT[3], color = :blue, markersize = 25, label = L" $x_T$")
axislegend(ax; 
        labelsize = 50, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (320.0,320.0,320.0,320.0))
fig
Makie.save("Bridges-drift.png", fig)

"""
    Sampling for 1 observation
"""
function ϕlogg(M, A, xT, ϕ)
    function out(t,x)
        i = Manifolds.get_chart_index(M, A, x)
        a = get_parameters(M, A, i, x)
        grad = ∇logg(M,induced_basis(M, A, i) , t , a , xT)
        # out = zeros(3)
        # Manifolds.get_vector_induced_basis!(M, out , x , dot(grad, ϕ(a)) , induced_basis(M, A, i) )
        return dot(grad, metric(M,induced_basis(M, A, i), a)*ϕ(a))
    end
    return out
end

function μ(M,A,X, xT, Φ)
    μ = zeros(length(Φ))
    for i in eachindex(Φ)
        for k in 1:length(X.tt)-1
            # println("tt = $(X.tt[k]) , μ = $(μ[i]), x = $(X.yy[k][1]), xT = $xT")
            μ[i] += ϕlogg(M,A,xT, Φ[i])(tt[k] , X.yy[k][1])*(X.tt[k+1] - X.tt[k])
        end
    end
    return μ
end


function μΓ(M,A,X, W, Φ::Array{T,1}) where {T<:Function}
    μ = zeros(length(Φ))
    Γ = zeros(length(Φ), length(Φ))
    for j in 1:length(X.tt)-1
        x, ν = X.yy[j]
        i = Manifolds.get_chart_index(M,A,x)
        B = Manifolds.induced_basis(M,A,i)
        a, Y = get_frame_parameterized(x,ν,M,B)
        Y⁻¹ = inv(Y)
        for k in eachindex(Φ)
            ϕk_param = zeros(length(a))
            Manifolds.get_coordinates_induced_basis!(M, ϕk_param, x, Φ[k](x) , B)
            μ[k] += dot( Y⁻¹*ϕk_param , W.yy[j+1] .- W.yy[j] )
            for ℓ in eachindex(Φ)
                ϕℓ_param = zeros(length(a))
                Manifolds.get_coordinates_induced_basis!(M,ϕℓ_param,x, Φ[ℓ](x) ,B)
                Γ[k,ℓ] +=  dot(Y⁻¹*ϕk_param, Y⁻¹*ϕℓ_param)*(W.tt[j+1] - W.tt[j])
            end
        end
    end
    return μ, Γ
end



θ₀ = [4.0, 4.0]
W = sample(tt, Wiener{SVector{2, Float64}}())
X = StochasticDevelopment(heun(), W, (M,B,t,a) -> V(θ₀, Φ)(a), (x₀,ν), M, A)
n = 20
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

function drift_multiple_obs(θ, obs)
    function out(M, B, t, a)
        k = getk(map(x -> x.t, obs), t)
        return V(θ, Φ)(a) .+ ∇logg(M,B,t,a,obs[k].u[1] )
    end
    return out
end


X = StochasticDevelopment(heun(), W, drift_multiple_obs([4.,4.],obs), (x₀, ν), M, A)

B = 100
ll = -1e10
Γ₀ = diagm([0.5, 0.5])
θ = rand(MvNormal(zeros(2), inv(Γ₀)))
# loglikelihoods = [-1e10 for i in 1:n]
W = sample(0:0.001:T, Wiener{SVector{2, Float64}}())
# BMs = separate(W,ind)
# bridges = separate(X, ind)
prog = Progress(B)
acc_W = 0
θ_array = [θ]
for j in 1:B
    # Update X given θ
    θ  = θ_array[end]
    for i in 1:n
        Wᵒ = crank_nicolson(0.75, BMs[i])
        drift(M,B,t,a) = V(θ, Φ)(a) .+ ∇logg( M,B,t,a,obs[i+1].u[1] )
        Xᵒ = StochasticDevelopment(heun(), Wᵒ, drift, obs[i].u, M, A)
        llᵒ = llikelihood(Xᵒ, obs[i+1].u[1], θ, Φ, M, A)
        if log(rand())  <= llᵒ - loglikelihoods[i] # accept
            loglikelihoods[i] = llᵒ
            BMs[i] = Wᵒ
            bridges[i] = Xᵒ
            acc_W += 1
        end
    end
    # Wᵒ  = crank_nicolson(0.7, W)
    # Xᵒ = StochasticDevelopment(heun(), Wᵒ, drift_multiple_obs(θ,obs), (x₀, ν), M, A)
    # llᵒ = llikelihood(Xᵒ, obs, θ, Φ, M, A)
    # if log(rand()) <= llᵒ - ll 
    #     ll = llᵒ
    #     W = Wᵒ
    #     X = Xᵒ
    # end
    # Update θ given X 
    X = concat(bridges)
    # ll = llikelihood(X, obs, θ, Φ, M, A)
    # bridgesᵒ = deepcopy(bridges)
    θᵒ = rand(MvNormal(θ, inv(Γ₀)))
    Xᵒ = StochasticDevelopment(heun(), W, drift_multiple_obs(θᵒ,obs), (x₀, ν), M, A)
    # loglikelihoodsᵒ = loglikelihoods
    # _μ = sum([μ(M, A, bridges[i], obs[i+1][1], Φ) for i in 1:n-1])
    # _μᵒ = zeros(2)
    # for i in 1:n
    #     drift(M,B,t,a) = V(θᵒ, Φ)(a) .+ ∇logg(M,B,t,a,obs[i+1].u[1])
    #     bridgesᵒ[i] = StochasticDevelopment(heun(), BMs[i], drift, obs[i].u, M, A)
    #     loglikelihoodsᵒ[i] = llikelihood(bridgesᵒ[i], obs[i+1].u[1], θᵒ, Φ, M, A)
    #     # _μᵒ += μ(M, A, bridgesᵒ[i], obs[i+1][1], Φ)
    # end
    llᵒ = llikelihood(Xᵒ, obs, θᵒ, Φ, M, A)
    if log(rand()) <=  llᵒ - ll # logpdf(MvNormal(inv(Γ₀)*_μᵒ, inv(Γ₀)), θᵒ) - logpdf(MvNormal(inv(Γ₀)*_μ, inv(Γ₀)), θ) 
        θ = θᵒ
        X = X
        ll = ll
    end
    _μ, Γ = μΓ(M,A,X,W,Φ)
    # println("Iteration $j: ")
    # println("μ = $_μ")
    # Γ += Γ₀
    # println("Γ = $Γ")
    # θ = inv(Γ)*_μ + sqrt(inv(Γ))*randn() #inv(Γ₀)*sum([ μ(M,A,bridges[i],obs[i+1][1],ϕ) for i in 1:n]) .+ inv(Γ₀)*randn() 
    push!(θ_array, θ)
    next!(prog)
end



fig = Figure(resolution=(2000, 2000), size = (1200,1200),fontsize=35)
ax1 = Axis(fig[1, 1],xlabel = "Iteration", ylabel = L"$\theta_1$")
Makie.lines!(ax1, collect(1:B+1) , map(x -> x[1], θ_array) ; linewidth = 3.0, color = palette(:default)[1], label = "Trace of sampler")
Makie.hlines!(ax1, [θ₀[1]] ; color =  palette(:default)[2], label = " True value", linewidth = 3.0)
axislegend(ax1; 
        labelsize = 35, 
        framewidth = 1.0, 
        labelfont = :bold, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (20.0,20.0,20.0,20.0))
ax2 = Axis(fig[2, 1],xlabel = "Iteration", ylabel = L"$\theta_2$")
Makie.lines!(ax2, collect(1:B+1) , map(x -> x[2], θ_array) ; linewidth = 3.0, color = palette(:default)[1], label = "Trace of sampler")
Makie.hlines!(ax2, [θ₀[2]] ; color =  palette(:default)[2], label = " True value", linewidth = 3.0)
axislegend(ax2; 
        labelsize = 35, 
        framewidth = 1.0, 
        labelfont = :bold, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (20.0,20.0,20.0,20.0))
fig

        Makie.save("trace.png", fig)

X = concat(bridges)
ax, fig = torus_figure(M)
lines!(ax, map(x -> x[1], map(x -> x[1], X.yy)), 
                map(x -> x[2], map(x -> x[1], X.yy)) , 
                map(x -> x[3], map(x -> x[1], X.yy)) ; 
                linewidth = 10.0, color = palette(:default)[1], label = L"$X")             
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
# Data generation, multiple observation times
θ₀ = [0.0, -4.0]
W = sample(tt, Wiener{SVector{2, Float64}}())
X = StochasticDevelopment(heun(), W, (M,B,t,a) -> V(θ₀, Φ)(a), (x₀,ν), M, A)
n = 20
ind = pushfirst!(Int64.(collect((length(tt)-1)/n:(length(tt)-1)/n:length(tt))), 1)
ind[end] = length(tt)
times = X.tt[ind]
obs = [observation(X.tt[i], X.yy[i]) for i in ind]

ax, fig = torus_figure(M)
lines!(ax, map(x -> x[1], map(x -> x[1], X.yy)), 
                map(x -> x[2], map(x -> x[1], X.yy)) , 
                map(x -> x[3], map(x -> x[1], X.yy)) ; 
                linewidth = 4.0, color = palette(:default)[1])
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


function concat(BMs::Array{T,1}) where {T<:SamplePath}
    tt = vcat(BMs[1].tt[1:end-1], BMs[2].tt)
    yy = vcat(BMs[1].yy[1:end-1], BMs[2].yy)
    for i in 3:length(BMs)
        tt = vcat(tt[1:end-1], BMs[i].tt)
        yy = vcat(yy[1:end-1], BMs[i].yy)
    end
    return SamplePath(tt,yy)
end
separate(X::SamplePath, ind) = [ SamplePath(X.tt[ind[i]:ind[i+1]] , X.yy[ind[i]:ind[i+1]]) for i in 1:length(ind)-1]


# Given α. update sample path
loglikelihoods = [-1e10 for i in 1:n]
loglikelihoodsᵒ = deepcopy(loglikelihoods)
total_ll = -1e10
θ = [0.0, 0.0]
σ = 1.0
θ_trace = [θ]
W = sample(0:0.001:T, Wiener{SVector{2, Float64}}())
BMs = separate(W,ind)
bridges = separate(X, ind)
bridgesᵒ = deepcopy(bridges)
B = 500
acc_θ = 0
acc_W = 0
prog = Progress(B)
for j in 1:B
    for i in 1:n
        tti = tt[ind[i]:ind[i+1]]
        Wᵒ = crank_nicolson(0.7, BMs[i])
        drift(M,B,t,a) = V(θ,Φ)(a) .+ ∇logg(M,B,t,a,obs[i+1][1])
        Xᵒ = StochasticDevelopment(heun(), Wᵒ, drift, obs[i], M, A)
        llᵒ = llikelihood(Xᵒ, obs[i+1][1], θ ,Φ, M, A)
        if log(rand())  <= llᵒ - loglikelihoods[i] # accept
            loglikelihoods[i] = llᵒ
            BMs[i] = Wᵒ
            bridges[i] = Xᵒ
            acc_W += 1
        end
    end
    total_ll = sum(loglikelihoods)
    W = concat(BMs)
    X = concat(bridges)
    # given W, X, update θ
    θᵒ = [θ[1],  θ[2] + σ*randn()]
    for i in 1:n
        drift(M,B,t,a) = V(θᵒ, Φ)(a) .+ ∇logg(M,B,t,a,obs[i+1][1])
        bridgesᵒ[i] = StochasticDevelopment(heun(), BMs[i], drift, obs[i], M, A)
        loglikelihoodsᵒ[i] = llikelihood(bridges[i], obs[i+1][1], θᵒ, Φ, M, A)
    end
    total_llᵒ = sum(loglikelihoodsᵒ)
    if log(rand()) <= total_llᵒ - total_ll
        θ = θᵒ
        total_ll = total_llᵒ
        loglikelihoods = deepcopy(loglikelihoodsᵒ)
        bridges = deepcopy(bridgesᵒ)
        acc_θ += 1
    end
    push!(θ_trace, θ)
    next!(prog)
end


fig = Figure(resolution=(2000, 1600), size = (1200,1200),fontsize=35)
ax = Axis(fig[1, 1],xlabel = "Iteration", ylabel = L"$\alpha$")
Makie.lines!(ax, collect(1:B+1) , map(x -> x[2], θ_trace) ; linewidth = 3.0, color = palette(:default)[1], label = "Trace of MH sampler")
Makie.hlines!(ax, [θ₀[2]] ; color =  palette(:default)[2], label = " True value", linewidth = 3.0)
axislegend(ax; 
        labelsize = 35, 
        framewidth = 1.0, 
        labelfont = :bold, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (20.0,20.0,20.0,20.0))
fig

X = concat(bridges)
ax, fig = torus_figure(M)
lines!(ax, map(x -> x[1], map(x -> x[1], X.yy)), 
                map(x -> x[2], map(x -> x[1], X.yy)) , 
                map(x -> x[3], map(x -> x[1], X.yy)) ; 
                linewidth = 4.0, color = palette(:default)[1])
Makie.scatter!(ax, obs[1][1][1], obs[1][1][2], obs[1][1][3], color = palette(:default)[2], markersize = 25, label = L" $x_i$")
for i in 2:n+1
    Makie.scatter!(ax, obs[i][1][1],obs[i][1][2],obs[i][1][3], color = palette(:default)[2], markersize = 25)
end
axislegend(ax; 
                labelsize = 50, 
                framewidth = 1.0, 
                orientation = :vertical,
                patchlabelgap = 18,
                patchsize = (50.0,50.0),
                margin = (320.0,320.0,320.0,320.0))
fig




nr_iterations = 500
θ = rand(MvNormal(zeros(2), inv(Γ₀)))
θ_array = [θ]
loglik = -1e10
prog = Progress(nr_iterations)
for k in 1:nr_iterations
    θᵒ = rand(MvNormal(θ, inv(Γ₀)))
    W = sample(tt, Wiener{SVector{2, Float64}}())
    BMs = separate(W, ind)
    ll = zeros(n)
    for i in 1:n
        tti = W.tt[ind[i]:ind[i+1]]
        drift(M,B,t,a) = V(θᵒ,Φ)(a) .+ ∇logg(M,B,t,a,obs[i+1].u[1])
        Xᵒ = StochasticDevelopment(heun(), BMs[i], drift, obs[i].u, M, A)
        ll[i] = llikelihood(Xᵒ, obs[i+1].u[1], θᵒ ,Φ, M, A)
    end
    loglikᵒ = sum(ll)
    if log(rand()) <= loglikᵒ - loglik
        θ = θᵒ
        # X = Xᵒ
        loglik = loglikᵒ
    end
    push!(θ_array, θ)
    next!(prog)
end
    
fig = Figure(resolution=(2000, 2000), size = (1200,1200),fontsize=35)
ax1 = Axis(fig[1, 1],xlabel = "Iteration", ylabel = L"$\theta_1$")
Makie.lines!(ax1, collect(1:nr_iterations+1) , map(x -> x[1], θ_array) ; linewidth = 3.0, color = palette(:default)[1], label = "Trace of sampler")
Makie.hlines!(ax1, [θ₀[1]] ; color =  palette(:default)[2], label = " True value", linewidth = 3.0)
axislegend(ax1; 
        labelsize = 35, 
        framewidth = 1.0, 
        labelfont = :bold, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (20.0,20.0,20.0,20.0))
ax2 = Axis(fig[2, 1],xlabel = "Iteration", ylabel = L"$\theta_2$")
Makie.lines!(ax2, collect(1:nr_iterations+1) , map(x -> x[2], θ_array) ; linewidth = 3.0, color = palette(:default)[1], label = "Trace of sampler")
Makie.hlines!(ax2, [θ₀[2]] ; color =  palette(:default)[2], label = " True value", linewidth = 3.0)
axislegend(ax2; 
        labelsize = 35, 
        framewidth = 1.0, 
        labelfont = :bold, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (20.0,20.0,20.0,20.0))
fig





θ_array = [[0.0, θ] for θ in -10.0:1.0:10.0]
loglik = zeros(length(θ_array))
# prog = Progress(length(θ_array)*n)
for (k,θ) in enumerate(θ_array)
    W = sample(tt, Wiener{SVector{2, Float64}}())
    BMs = separate(W, ind)
    # ll = zeros(n)
    _μ = [zeros(2) for i in 1:n]
    for i in 1:n
        tti = W.tt[ind[i]:ind[i+1]]
        drift(M,B,t,a) = V(θ,Φ)(a) .+ ∇logg( M,B,t,a,obs[i+1].u[1] )
        X = StochasticDevelopment(heun(), BMs[i], drift, obs[i].u, M, A)
        _μ[i] += μ(M,A,X,obs[i].u[1], Φ)
        # ll[i] =  llikelihood(X, obs[i+1].u[1], θ ,Φ, M, A)
        # next!(prog)
    end
    println("Iteration $k of $(length(θ_array));  μ = $(sum(_μ))")
    loglik[k] += logpdf(MvNormal(inv(Γ₀)*sum(_μ), inv(Γ₀)) , θ) # sum(ll)
end

θ_arr = map(t -> t[2], θ_array)
fig = Figure(resolution=(2000, 1600), size = (1200,1200),fontsize=35)
ax = Axis(fig[1, 1],xlabel = L"$\theta$", ylabel = L"$\log\, L(\theta)")
Makie.lines!(ax, θ_arr[map(b->!isnan(b) && !isinf(b), loglik)], loglik[map(b->!isnan(b) && !isinf(b), loglik)] ; linewidth = 3.0, color = palette(:default)[1], label = " Loglikelihood")
Makie.vlines!(ax, [θ₀[2]] ; color = :red, label = " True value", linewidth = 3.0)
axislegend(ax; 
        labelsize = 35, 
        framewidth = 1.0, 
        labelfont = :bold, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (20.0,20.0,20.0,20.0))
fig
Makie.save("likelihood_estimation.png", fig)






# Data generation Independent observations of xT
dt = 0.001
TimeChange(T) = (x) ->  x * (2-x/T)
tt = 0.:dt:T # TimeChange(T).(0.:dt:T)

α₀ = 4.0
n = 20
pts = []
for k in 1:n
    W = sample(tt, Wiener{SVector{2, Float64}}())
    X = StochasticDevelopment(heun(), W, (M,B,t,a) -> V(M,B,a, α₀), (x₀,ν), M, A)
    push!(pts, X.yy[end][1])
end
ax, fig = torus_figure(M)
Makie.scatter!(ax, x₀[1],x₀[2],x₀[3], color = :red, markersize = 25, label = L" $x_0$")
Makie.scatter!(ax, pts[1][1], pts[1][2], pts[1][3], color = :blue, markersize = 25, label = L" $x_i$")
for k in 2:n
    Makie.scatter!(ax, pts[k][1], pts[k][2], pts[k][3], color = :blue, markersize = 25)
end
axislegend(ax; 
        labelsize = 50, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (320.0,320.0,320.0,320.0))
fig


α_array = collect(-10.0:1.0:10.0)
loglik = zeros(length(α_array))
prog = Progress(length(α_array)*n)
for (k,α) in enumerate(α_array)
    for i in 1:n
        W = sample(tt, Wiener{SVector{2, Float64}}())
        drift(M,B,t,a) = V(M,B,a, α) .+ ∇logg(M,B,t,a,pts[i])
        StochasticDevelopment!(heun(), X, W, drift, (x₀,ν), M, A)
        loglik[k] += llikelihood(X, pts[i], α, M,A)
        next!(prog)
    end
end

fig = Figure(resolution=(2000, 1600), size = (1200,1200),fontsize=35)
ax = Axis(fig[1, 1],xlabel = L"$\alpha$", ylabel = L"$\ell(\alpha)")
Makie.lines!(ax, α_array[map(b->!isnan(b) && !isinf(b), loglik)], loglik[map(b->!isnan(b) && !isinf(b), loglik)] ; linewidth = 3.0, color = palette(:default)[1], label = " Loglikelihood")
Makie.vlines!(ax, [α₀] ; color = :red, label = " True value", linewidth = 3.0)
axislegend(ax; 
        labelsize = 35, 
        framewidth = 1.0, 
        labelfont = :bold, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (20.0,20.0,20.0,20.0))
fig




ax, fig = torus_figure(M)
k = 1
for α in [-6.0, -3.0, 0.0, 3.0, 6.0]
    W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
    drift(M,B,t,a) = V(M,B,a, α) .+ ∇logg(M,B,t,a,xT)
    X = StochasticDevelopment(heun(), W, drift, (x₀, ν), M, A)
    lines!(ax, map(x -> x[1][1], X.yy), 
                map(x -> x[1][2], X.yy) , 
                map(x -> x[1][3], X.yy) ; 
                linewidth = 4.0,  color = palette(:default)[k],label = "α = $α")
    k += 1
end
Makie.scatter!(ax, x₀[1],x₀[2],x₀[3], color = :red, markersize = 25, label = L" $x_0$")
Makie.scatter!(ax, xT[1],xT[2],xT[3], color = :blue, markersize = 25, label = L" $x_T$")
axislegend(ax; 
        labelsize = 50, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (200.0,200.0,200.0,200.0))
fig

######
# Sphere
#####

M = Manifolds.Sphere(2)
A = Manifolds.StereographicAtlas()

ax1, fig1 = sphere_figure()
# lines!(ax1, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 2.0, color = :green)
fig1
Makie.save("sphere.png", fig1)

p = [0.0,0.0,1.0]
i = Manifolds.get_chart_index(M, A,p)
B = induced_basis(M,A,i)
N = p # normal vector
ν = nullspace(N')

a, Y = get_frame_parameterized(p,ν,M,B)

function christoffel_symbols_second(M::Manifolds.Sphere, B::AbstractBasis, p)
    Γ = zeros(2,2,2)
    u,v = Manifolds.get_parameters(M, B.A, B.i, p)
    den = 1+u^2+v^2
    Γ[1,:,:] = (2/den) .* [-u -v ; -v u]
    Γ[2,:,:] = (2/den) .* [v -u ; -u -v]
    return Γ
end



tt = 0.0:0.001:10.0
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
γ = deepcopy(W) ; for i in eachindex(γ.tt) γ.yy[i] = 1.0.*[γ.tt[i], 2*γ.tt[i]] end
X = StochasticDevelopment(heun(), W, (p, ν), M, A)
X = HorizontalDevelopment(heun(), γ, (p,ν), M, A)
xx = map(x -> x[1], X.yy)

fig = Figure(resolution=(2000, 1600), size = (1200,1200),fontsize=46)
ax = Axis(fig[1, 1], title = "Brownian motion in ℝ²", xlabel = "x", ylabel = "y")
Makie.lines!(ax, map(x -> x[1], γ.yy), map(x -> x[2], γ.yy) ; linewidth = 2.0, color = palette(:default)[1])
fig
Makie.save("BMR2.png", fig)

ax1, fig1 = sphere_figure()
lines!(ax1, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 2.0, color = palette(:default)[1])
Label(fig1[1,1,Top()], "Brownian motion in 𝕊²")
fig1
Makie.save("BMS2.png", fig1)



K = 50
function κ(t, y , z, M::Manifolds.Sphere, B::AbstractBasis)
    yp = Manifolds.get_point(M, B.A, B.i, y)
    zp = Manifolds.get_point(M, B.A, B.i, z)
    sum([ exp(-k*(k+1)*t/2)*(2*k+1)/(2*pi)*LegendrePolynomials.Pl(dot(yp,zp),k) for k in 0:1:K ])
end

T = 1.0
v = [0.0, 0.0,-1.0]

check_point(M,v)

g(t,a, M::Manifolds.Sphere, B::AbstractBasis) = κ(T-t, a, Manifolds.get_parameters(M,B.A,B.i,v), M, B)

function cometric(a, M, B::AbstractBasis)
    u,v = a[1], a[2]
    return 0.25*(1+u^2+v^2)*I
end

function ∇logg(t, a, M::Manifolds.Sphere, B::AbstractBasis)
    _∇ = ForwardDiff.gradient(a -> log(g(t,a, M, B)), a)
    g⁺ = cometric(a, M, B)
    return g⁺*_∇
end



T = 1.0
tt = 0.0:0.001:T
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
X = StochasticDevelopment_drift(heun(), W, (p, ν), M, A)
xx = map(x -> x[1], X.yy)

ax, fig = sphere_figure()
lin = lines!(ax, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 2.0, color = :green, label = "Xₜ")
x0_pt  = Makie.scatter!(ax, p[1],p[2],p[3], color = :red, markersize = 25, label = "x₀")
xT_pt = Makie.scatter!(ax, v[1],v[2],v[3], color = :blue, markersize = 25, label = "xT")
axislegend(ax; labelsize = 50)
fig





