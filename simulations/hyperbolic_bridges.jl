include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")
using Random
using DelimitedFiles
using Plots
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
    Simulating Brownian motion in local coordinates using the Laplace Beltrami operator

W = sample(tt, Wiener{SVector{2,Float64}}())
A = [a₀]
for i in 1:length(tt)-1
    push!(A, A[end] + 0.5*(1-dot(A[end],A[end]))*(W.yy[i+1] .- W.yy[i]) )
end

CairoMakie.activate!()
fig = let
    f = Figure()
    Axis(f[1,1])
    arc!(Point2f(0), 1, -π, π, color = :red)
    Makie.lines!(map(x -> x[1], A), map(x -> x[2], A))
    f
end
"""

"""
    Simulation of Brownian motion using stochastic HorizontalDevelopment
"""
W = sample(tt, Wiener{SVector{2,Float64}}())
drift(M,t,a) = [0.0, 0.0]
U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
aa = map(x -> convert(PoincareBallPoint, x).value, xx)

CairoMakie.activate!()
fig = let
    f = Figure()
    Axis(f[1,1])
    arc!(Point2f(0), 1, -π, π, color = :red)
    Makie.lines!(map(x -> x[1], aa), map(x -> x[2], aa))
    f
end
display(fig)

GLMakie.activate!()
fig = let 
    ax, fig = hyperboloid_figure(M)
    Makie.lines!(map(x -> x[1], xx),map(x -> x[2], xx),map(x -> x[3], xx))
    fig
end
display(fig)

"""
    Bridge Simulation
"""
ρs = 0.0:0.001:10.0
grid = gridρΔ(ρs,tt)
fill_grid!(grid, 10000, (ρ,t) -> ρ+2*sqrt(t))

writedlm(outdir*"hyperbolic_grid_filled_v2.csv",grid.grid_vals,',')

xT = xx[end]; νT = νν[end]
aT, YT = get_frame_parameterized(xT, νT, M)
obs = observation(T, (xT, νT))
drift(M,t,a) = ∇logg(M, t, a, grid, obs)

W = sample(tt, Wiener{SVector{2, Float64}}())
U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
xx = map(u -> u[1], U.yy)
aa = map(x -> convert(PoincareBallPoint, x).value, xx)

CairoMakie.activate!()
fig = let
    f = Figure(resolution=(800, 800), size = (1200,1200), fontsize=20)
    ax = Axis(f[1,1])
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
    f
end
Makie.save(outdir*"bridge_poincare_disk2.png", fig)

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
V(a) = -5*a

CairoMakie.activate!()
fig = let
    f = Figure(resolution=(800, 800), size = (1200,1200), fontsize=20)
    ax = Axis(f[1,1])
    pts = [0.85 .* [cos(φ), sin(φ)] for φ ∈ range(0,2π,length=11)]
    arc!(Point2f(0), 1, -π, π, color = :red)
    Makie.scatter!(ax, map(p -> p[1], pts), map(p -> p[2], pts), markersize = 15)
    Makie.arrows!(ax, map(p -> p[1], pts),map(p -> p[2], pts),
                            map(p -> V(p)[1], pts),map(p -> V(p)[2], pts) , color = palette(:default)[2])
    # axislegend(ax; 
    #     labelsize = 35, 
    #     framewidth = 1.0, 
    #     orientation = :vertical,
    #     patchlabelgap = 10,
    #     patchsize = (40.0,40.0),
    #     margin = (32.0,32.0,32.0,32.0))
    fig
    f
end
Makie.save(outdir*"vector_field_hyperbolic.png", fig)


W = sample(tt, Wiener{SVector{2,Float64}}())
drift(M,t,a) = V(a)
U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
aa = map(x -> convert(PoincareBallPoint, x).value, xx)

xT = xx[end]; νT = νν[end]
aT, YT = get_frame_parameterized(xT, νT, M)
obs = observation(T, (xT, νT))

CairoMakie.activate!()
fig = let
    f = Figure(resolution=(800, 800), size = (1200,1200), fontsize=20)
    ax = Axis(f[1,1])
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
    f
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



# Bridge simulation

W = sample(tt, Wiener{SVector{2,Float64}}())
drift(M,t,a) = V(a) + ∇logg(M,t,a,grid,obs)
U = StochasticDevelopment(heun(), W, drift, (x₀,ν₀), M)
xx = map(u -> u[1], U.yy) ; νν = map(u -> u[2], U.yy)
aa = map(x -> convert(PoincareBallPoint, x).value, xx)

CairoMakie.activate!()
fig = let
    f = Figure(resolution=(800, 800), size = (1200,1200), fontsize=20)
    ax = Axis(f[1,1])
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
    f
end
display(fig)
Makie.save(outdir*"bridge_drift_poincare_disk2.png", fig)

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
