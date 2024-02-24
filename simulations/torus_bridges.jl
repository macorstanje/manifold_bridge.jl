include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")
GLMakie.activate!()
using Random
using ProgressMeter
outdir = "/Users/marc/Documents/Onderzoek/ManifoldGPs/"
Random.seed!(61)

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
xx = map(x -> x[1], X.yy)

GLMakie.activate!()
fig = let
    ax, fig = torus_figure(M)
    lines!(ax, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 2.0, color = palette(:default)[1])
    fig
end


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

fig = let
    ax, fig = torus_figure(M)
    lines!(ax, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 15.0, color = :green)
    arrows!(ax, pts_array, vec1_array, linewidth=0.04, color=:blue, transparency = false)
    arrows!(ax, pts_array, vec2_array, linewidth=0.04, color=:blue, transparency = false)
    fig
end
# Makie.save("horizontal_curve.png", fig)

# Plot initial frame
fig = let
    ax, fig = torus_figure(M)
    Makie.scatter!(ax, x₀[1],x₀[2],x₀[3], color = :red, markersize = 45, label = "x₀")
    arrows!(ax, [Point3f(xx[1])], [Point3f(2.0.*νν[1][:,1])], linewidth=0.04, color=:red, transparency = false)
    arrows!(ax, [Point3f(xx[1])], [Point3f(2.0.*νν[1][:,2])], linewidth=0.04, color=:red, transparency = false)
    fig
end
# Makie.save("frame.png", fig)

"""
    Brownian bridge simulation
"""
a₀ = Manifolds.get_parameters(M, B.A,B.i,x₀)
# Simulation, conditioned to hit xT
T = 1.0
xT = [-3.0, 0.0,2.0]
obs = observation(T, Frame(xT))

tt = 0.0:0.001:T
drift(M,B,t,a) = ∇logg(M,B,t,a, obs)
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
fig = let
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
end
Makie.save(outdir*"standard_BB_no_vector_field.png", fig)


"""
    Bridge process with a vector field
"""

# Definition of the vector field in terms of a parameter θ
function linear_combination(θ, ϕ::Array{T, 1}) where {T<:Function}
    K = typeof(θ) <: Real ? 1 : length(θ)
    return x -> K == 1 ? θ*ϕ[1](x) : sum([θ[k]*ϕ[k](x) for k in 1:K])
end
ϕ1(a) = [1., 0.]
ϕ2(a) = [0., 1.]
Φ = [ϕ1, ϕ2]
V(θ, Φ) = linear_combination(θ, Φ)


# Illustration of the vector field
θ₀ = [0.0,-4.0]
fig = let
    N = 20
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
end
Makie.save(outdir*"vector_field_40.png", fig)

# Bridge process, conditioned to hit xT
T = 1.0
tt = 0.0:0.001:T
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
obs = observation(T, Frame(xT))
drift(M,B,t,a) = V(θ₀, Φ)(a) .+ ∇logg(M,B,t,a,obs)
X = StochasticDevelopment(heun(), W, drift, (x₀, ν), M, A)

# multiple samplepaths
xx = map(x -> x[1], X.yy)
samplepaths = [xx]
for k in 1:10
    W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
    StochasticDevelopment!(heun(), X, W, drift, (x₀, ν), M, A)
    push!(samplepaths, map(x -> x[1], X.yy))
end

fig = let
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
end
Makie.save(outdir*"bridges_vector_field_0-4.png", fig)





"""
    Bridge simulation using Crank-Nicolson
"""
# Forward Trajectory
T = 1.0
tt = map((x) ->  x * (2-x/T) , 0.0:0.001:T)
θ₀ = [2.0, 2.0]
drift(M,B,t,a) = V(θ₀, Φ)(a)
W = sample(tt, Wiener{SVector{2, Float64}}())
X = StochasticDevelopment(heun(), W, drift,(x₀,ν), M, A)

GLMakie.activate!()
fig = let
    ax, fig = torus_figure(M)
    lines!(ax, map(x -> x[1][1], X.yy), 
            map(x -> x[1][2], X.yy) , 
            map(x -> x[1][3], X.yy) ; 
            linewidth = 4.0, color = palette(:default)[1])
    Makie.scatter!(ax, x₀[1],x₀[2],x₀[3], color = :red, markersize = 25, label = L" $x_0$")
    Makie.scatter!(ax, X.yy[end][1][1], X.yy[end][1][2], X.yy[end][1][3], color = :blue, markersize = 25, label = L" $x_T$")
    axislegend(ax; 
            labelsize = 50, 
            framewidth = 1.0, 
            orientation = :vertical,
            patchlabelgap = 18,
            patchsize = (50.0,50.0),
            margin = (320.0,320.0,320.0,320.0))
    fig
end
Makie.save(outdir*"Forward_trajectory_22.png", fig)

# Bridge to endpoint of forward observation
xT = X.yy[end][1]
obs = observation(T, Frame(xT))
drift(M,B,t,a) = V(θ₀, Φ)(a)  .+ ∇logg(M,B,t,a,obs)
W = sample(tt, Wiener{SVector{2, Float64}}())
X = StochasticDevelopment(heun(), W, drift, (x₀, ν), M, A)
Xᵒ = deepcopy(X)
ll = loglikelihood(X, obs, θ₀, Φ, M, A)

# Runs of the Crank-Nicolson scheme
nr_simulations = 1000
xx = map(x -> x[1], X.yy)
samplepaths = [xx]
acc = [0]
prog = Progress(nr_simulations)
for k in 1:nr_simulations
    λ = .9
    Wᵒ = crank_nicolson(λ, W)
    StochasticDevelopment!(heun(), Xᵒ, Wᵒ, drift, (x₀, ν), M, A)
    llᵒ = loglikelihood(Xᵒ, obs, θ₀, Φ, M, A)
    if log(rand()) <= llᵒ - ll
        push!(acc, acc[end] + 1)
        W = Wᵒ
        X = Xᵒ
        ll = llᵒ
        push!(samplepaths, map(x -> x[1], X.yy))
    else
        push!(acc, acc[end])
    end
    next!(prog)
end

# Plot of the behavior of the acceptance rate (should be increasing)
CairoMakie.activate!()
fig = let 
    fig = Figure(resolution=(2000, 1000), size = (1200,1200),fontsize=35)
    ax1 = Axis(fig[1, 1],xlabel = "Iteration", ylabel = "accepted proposals")
    Makie.lines!(ax1, 1:nr_simulations, acc[2:end] ; linewidth = 3.0, color = palette(:default)[1])
    fig
end

GLMakie.activate!()
fig = let
    ax, fig = torus_figure(M)
    for k in 1:6
        lines!(ax, map(x -> x[1], samplepaths[end-k+1]), 
                    map(x -> x[2], samplepaths[end-k+1]) , 
                    map(x -> x[3], samplepaths[end-k+1]) ; 
                    linewidth = 4.0, color = palette(:oslo100)[Int64(round(k*length( palette(:oslo100))/10))],
        )# label = "Iteration $(length(samplepaths)-k+1)")
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
end
Makie.save(outdir*"Bridges-drift_CN_after_forward_trajectory_22.png", fig)