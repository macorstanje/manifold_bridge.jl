include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")
using Random
using ProgressMeter
using StaticArrays
using Plots
using DelimitedFiles
outdir = "/Users/marc/Documents/Onderzoek/ManifoldGPs/"
Random.seed!(61)

"""
    Initial settings
"""

M = Hyperbolic(2)
# Starting point
a₀ = [0, 0.4] 
Y₀ = (1-dot(a₀,a₀))/2*[1 0 ; 0 1]
x₀, ν₀ = get_frame_vectors(a₀,Y₀,M)
# time Scale
T = 1.0
tt = 0.0:0.001:T

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


"""
    Simulation of Brownian motion using stochastic HorizontalDevelopment
"""
W = sample(tt, Wiener{SVector{2,Float64}}())
drift(M,t,a) = [0.0, 0.0]
U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
aa = map(x -> convert(PoincareBallPoint, x).value, xx)
writedlm(outdir*"trajectory.csv", aa, ',')

samplepaths = [xx]
for i in 1:5
    W = sample(tt, Wiener{SVector{2,Float64}}())
    U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
    push!(samplepaths, map(u -> u[1], U.yy))
end

CairoMakie.activate!()
fig = let
    f = Figure()
    Axis(f[1,1])
    arc!(Point2f(0), 1, -π, π, color = :red)
    for (i,xx) in enumerate(samplepaths)
        aa = map(x -> convert(PoincareBallPoint, x).value, xx)
        Makie.lines!(map(x -> x[1], aa), map(x -> x[2], aa), color = color = palette(:default)[i])
    end
    Makie.scatter!(a₀[1], a₀[2], markersize = 20, color = "red")
    f
end
display(fig)
Makie.save(outdir*"forward_paths_hyperbolic.png", fig)

GLMakie.activate!()
fig = let 
    ax, fig = hyperboloid_figure(M)
    Makie.lines!(map(x -> x[1], xx),map(x -> x[2], xx),map(x -> x[3], xx))
    fig
end
display(fig)

"""
    Brownian bridge Bridge Simulation
"""
# Simulating Brownian motion
W = sample(tt, Wiener{SVector{2, Float64}}())
drift(M,t,a) = zero(a)
X = StochasticDevelopment(heun(), W, drift, (x₀, ν₀) , M)
xx = map(x -> x[1], X.yy) # global coordinates
aa = map(x -> convert(PoincareBallPoint, x).value, xx) # local coordinates

# Select the end point of the forward trajectory as end point
xT = xx[end]; νT = νν[end]
aT, YT = get_frame_parameterized(xT, νT, M)
obs = observation(T, (xT, νT))

Z = randn(5000)
Zpos = Z[Z.>-2]
drift(M,t,a) = ∇logg(M, t, a, obs, Zpos)

# Figure on Poincare disk
CairoMakie.activate!()
fig = let
    fig = Figure()
    ax = Axis(fig[1, 1], yautolimitmargin = (0.1, 0.1), xautolimitmargin = (0.1, 0.1))
    arc!(Point2f(0), 1, -π, π, color = :red)
    Makie.lines!(map(x -> x[1], aa), map(x -> x[2], aa), color = color = palette(:default)[1])
    Makie.scatter!(a₀[1],a₀[2], color = :red, label = L"$x_0$", markersize = 20)
    Makie.scatter!(aT[1], aT[2], color = :blue, label = L"$x_T$", markersize = 20)
    axislegend(ax; 
        labelsize = 35, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 10,
        patchsize = (40.0,40.0),
        margin = (32.0,32.0,32.0,32.0))
    fig
end
Makie.save(outdir*"bridge_poincare_disk2.png", fig)

# Figure on hyperboloid
GLMakie.activate!()
fig = let 
    ax, fig = hyperboloid_figure(M)
    Makie.lines!(map(x -> x[1], xx),map(x -> x[2], xx),map(x -> x[3], xx), color = palette(:default)[1])
    Makie.scatter!(x₀[1],x₀[2],x₀[3], color = :red, label = L"$x_0$", markersize = 20)
    Makie.scatter!(xT[1], xT[2],xT[3], color = :blue, label = L"$x_T$", markersize = 20)
    axislegend(ax; 
        labelsize = 50, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (152.0,152.0,152.0,152.0))
    fig
end
Makie.save(outdir*"bridge_hyperbolic2.png", fig)


"""
    With a drift
"""
# Visualization of the drift
V(θ) = (a) -> 0.3*[θ*(1-dot(a,a)), 0] # Scale down by 0.3 for reasonable visualization
CairoMakie.activate!()
fig = let
    f = Figure(resolution=(900, 800), size = (1200,1200), fontsize=20)
    ax = Axis(f[1,1])
    pts = [0.85 .* [cos(φ), sin(φ)] for φ ∈ range(0,2π,length=11)]
    arc!(Point2f(0), 1, -π, π, color = :red)
    Makie.scatter!(ax, map(p -> p[1], pts), map(p -> p[2], pts), markersize = 15)
    Makie.arrows!(ax, map(p -> p[1], pts),map(p -> p[2], pts),
                            map(p -> V(p)[1], pts),map(p -> V(p)[2], pts) , color = palette(:default)[2])
    fig
    f
end
Makie.save(outdir*"vector_field_hyperbolic_righ2t.png", fig)

# Forward simulation
W = sample(tt, Wiener{SVector{2,Float64}}())
drift(M,t,a) = V(5)(a)
U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
aa = map(x -> convert(PoincareBallPoint, x).value, xx)

#xT = xx[end]; νT = νν[end]
xT = [x₀[1], -x₀[2], x₀[3]] 
νT = [-ν₀[1,1] ν₀[1,2] ; ν₀[2,1] -ν₀[2,2] ; ν₀[3,1] -ν₀[3,2]]
aT, YT = get_frame_parameterized(xT, νT, M)
obs = observation(T, (xT, νT))

CairoMakie.activate!()
fig = let
    fig = Figure()
    ax = Axis(fig[1, 1], yautolimitmargin = (0.1, 0.1), xautolimitmargin = (0.1, 0.1))
    arc!(Point2f(0), 1, -π, π, color = :red)
    Makie.lines!(map(x -> x[1], aa), map(x -> x[2], aa), color = palette(:default)[1])
    Makie.scatter!(a₀[1],a₀[2], color = :red, label = L"$x_0$", markersize = 20)
    Makie.scatter!(aT[1], aT[2], color = :blue, label = L"$x_T$", markersize = 20)
    axislegend(ax; 
        labelsize = 35, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 10,
        patchsize = (40.0,40.0),
        margin = (32.0,32.0,32.0,32.0))
    fig
end
display(fig)
# Makie.save(outdir*"bridge_poincare_disk.png", fig)

GLMakie.activate!()
fig = let
    ax, fig = hyperboloid_figure(M)
    Makie.lines!(map(x -> x[1], xx),map(x -> x[2], xx),map(x -> x[3], xx), color = palette(:default)[1])
    Makie.scatter!(x₀[1],x₀[2],x₀[3], color = :red, label = L"$x_0$", markersize = 20)
    Makie.scatter!(xT[1], xT[2],xT[3], color = :blue, label = L"$x_T$", markersize = 20)
    axislegend(ax; 
        labelsize = 50, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (152.0,152.0,152.0,152.0))
    fig
end
display(fig)
# Makie.save(outdir*"bridge_hyperbolic.png", fig)


"""
    Use this part for bridge simulaton with a vector field!
"""
# Select the opposing point as end point
a₀ = [0.6,0.0]
Y₀ = (1-dot(a₀,a₀))/2*[1 0 ; 0 1]
x₀, ν₀ = get_frame_vectors(a₀,Y₀,M)


xT = [x₀[1], -x₀[2], x₀[3]] 
νT = [-ν₀[1,1] ν₀[1,2] ; ν₀[2,1] -ν₀[2,2] ; ν₀[3,1] -ν₀[3,2]]
aT, YT = get_frame_parameterized(xT, νT, M)
obs = observation(T, (xT, νT))

# Inward pointing vector field
V(θ)(a) = θ*a
W = sample(tt, Wiener{SVector{2,Float64}}())
drift(M,t,a) = V(-15)(a) + ∇logg(M,t,a,obs, Zpos)
U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
aa = map(x -> convert(PoincareBallPoint, x).value, xx)

# Visualizing the guiding term and distance to end point along the path
guiding_term = [∇logg(M,U.tt[i],aa[i],obs, Zpos) for i in 1:length(aa)-1]
distan = [dist(M, a, convert(PoincareBallPoint, HyperboloidPoint(obs.u[1])).value) for a in aa]

# Plot the guiding term
CairoMakie.activate!()
fig = let 
    fig = Figure()
    ax = Axis(fig[1, 1], yautolimitmargin = (0.1, 0.1), xautolimitmargin = (0.1, 0.1))
    Makie.lines!(U.tt[1:end-1], map(a -> a[1], guiding_term), label = L"$\nabla \log g_1$")
    Makie.lines!(U.tt[1:end-1], map(a -> a[2], guiding_term), label = L"$\nabla \log g_2$")
    axislegend(ax; labelsize = 25, patchsize = (50.0,30.0), position = :rb)
    fig
end
Makie.save(outdir*"guiding_term.png", fig)

# Plot the distance from X_t to x_T
CairoMakie.activate!()
fig = let 
    fig = Figure()
    ax = Axis(fig[1, 1], yautolimitmargin = (0.1, 0.1), xautolimitmargin = (0.1, 0.1))
    Makie.lines!(U.tt, distan, label = L"$\rho$")
    ax.xlabel = "t"
    ax.ylabel = L"$\rho$"
    fig
end

# Plot the process in the poincaredisk
CairoMakie.activate!()
fig = let
    fig = Figure()
    ax = Axis(fig[1, 1], yautolimitmargin = (0.1, 0.1), xautolimitmargin = (0.1, 0.1))
    arc!(Point2f(0), 1, -π, π, color = :red)
    Makie.lines!(map(x -> x[1], aa[1:end-3]), map(x -> x[2], aa[1:end-3]), color = color = palette(:default)[1])
    Makie.scatter!(a₀[1],a₀[2], color = :red, label = L"$x_0$", markersize = 20)
    Makie.scatter!(aT[1], aT[2], color = :blue, label = L"$x_T$", markersize = 20)
    axislegend(ax; 
        labelsize = 35, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 10,
        patchsize = (40.0,40.0),
        margin = (32.0,32.0,32.0,32.0))
    fig
end


# Plot the process on on an hyperboloid
GLMakie.activate!()
fig = let
    ax, fig = hyperboloid_figure(M)
    Makie.lines!(map(x -> x[1], xx),map(x -> x[2], xx),map(x -> x[3], xx), color = palette(:default)[1])
    Makie.scatter!(x₀[1],x₀[2],x₀[3], color = :red, label = L"$x_0$", markersize = 20)
    Makie.scatter!(xT[1], xT[2],xT[3], color = :blue, label = L"$x_T$", markersize = 20)
    axislegend(ax; 
        labelsize = 50, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (152.0,152.0,152.0,152.0))
    fig
end
display(fig)
Makie.save(outdir*"bridge_drift_hyperbolic2.png", fig)


# Crank nicolson scheme
X = deepcopy(U)
nr_simulations = 50
xx = map(x -> x[1], X.yy)
samplepaths = [xx]
acc = [true]
Xᵒ = deepcopy(X)
ll = loglikelihood(Xᵒ, obs, M, Zpos)
llvals = [ll]
prog = Progress(nr_simulations)
for k in 1:nr_simulations
    λ = .95
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

# Plot 5 accepted samplepaths (evenly spread) on poicare disk
CairoMakie.activate!()
fig = let
    fig = Figure()
    ax = Axis(fig[1, 1], yautolimitmargin = (0.1, 0.1), xautolimitmargin = (0.1, 0.1))
    arc!(Point2f(0), 1, -π, π, color = :red)
    accepted_samplepaths = samplepaths[acc]
    samplepaths_to_plot = accepted_samplepaths[[Int(ceil(0.2*k*sum(acc))) for k in 1:5]]
    for (i,xx) in enumerate(samplepaths_to_plot)
        aa = map(x -> convert(PoincareBallPoint, x).value, xx)
        Makie.lines!(map(x -> x[1], aa), map(x -> x[2], aa), color = color = palette(:oslo50)[1+10*(i-1)])
    end
    Makie.scatter!(a₀[1],a₀[2], color = :red, label = L"$x_0$", markersize = 20)
    Makie.scatter!(aT[1], aT[2], color = :blue, label = L"$x_T$", markersize = 20)
    axislegend(ax; 
        labelsize = 35, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 10,
        patchsize = (40.0,40.0),
        margin = (32.0,32.0,32.0,32.0))
    fig
end
Makie.save(outdir*"bridge_drift_hyperbolic3_no_field.png", fig)

# Plot 5 accepted samplepaths (evenly spread) on hyperboloid
GLMakie.activate!()
fig = let
    ax, fig = hyperboloid_figure(M)
    accepted_samplepaths = samplepaths[acc]
    samplepaths_to_plot = accepted_samplepaths[[Int(ceil(0.2*k*sum(acc))) for k in 1:5]]
    for (i,xx) in enumerate(samplepaths_to_plot)
        Makie.lines!(map(x -> x[1], xx),map(x -> x[2], xx),map(x -> x[3], xx), color =  palette(:oslo50)[1+10*(i-1)])
    end
    Makie.scatter!(x₀[1],x₀[2],x₀[3], color = :red, label = L"$x_0$", markersize = 20)
    Makie.scatter!(xT[1], xT[2],xT[3], color = :blue, label = L"$x_T$", markersize = 20)
    axislegend(ax; 
        labelsize = 50, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (152.0,152.0,152.0,152.0))
    fig
end
display(fig)
Makie.save(outdir*"bridge_drift_hyperbolic4.png", fig)

# Make a traceplot of the endpoint
CairoMakie.activate!()
fig = let 
    fig = Figure()
    ax = Axis(fig[1, 1], yautolimitmargin = (0.1, 0.1), xautolimitmargin = (0.1, 0.1))
    accepted_samplepaths = samplepaths[acc]
    ind = Int(ceil(1.0*length(X.tt))) # or select another index here
    aa = []
    for xx in accepted_samplepaths
        push!(aa, convert(PoincareBallPoint, xx[ind]).value)
    end
    Makie.lines!(1:sum(acc) , map(a -> a[1], aa), label = L"$a_1$")
    Makie.lines!(1:sum(acc) , map(a -> a[2], aa), label = L"$a_2$")
    Makie.hlines!(aT[1], linewidth = 1)
    Makie.hlines!(aT[2], linewidth = 1)
    axislegend(ax; labelsize = 25, patchsize = (50.0,30.0), position = :lt)
    fig
end
Makie.save(outdir*"trace_plot_no_field_end_point.png", fig)