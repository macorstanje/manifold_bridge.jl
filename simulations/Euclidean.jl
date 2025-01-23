include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")
using Random
using ProgressMeter
using MakiePublication
outdir = "/Users/marc/Documents/Onderzoek/ManifoldGPs/"

M = Manifolds.Euclidean(2)

x₀ = [0.,0.]
ν = [1. 0. ; 0. 1. ]

T = 5.0
tt = 0.0:0.001:T
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())


θ = 2*pi*tt
a = 0.6
r = a .*θ
xx = r .* cos.(θ)+map(x -> x[1], W.yy)
yy = r .* sin.(θ)+map(x -> x[2], W.yy)

CairoMakie.activate!()
function plot_process()
    f = Figure(resolution=(3000, 3000), size = (5000,3000))
    ax = Axis(f[1,1], show_axis = false , show_grid = false)
    Makie.lines!(ax, xx , yy, linewidth = 3.0, color = palette(:nuuk25)[1])
    hidespines!(ax) ; hidedecorations!(ax)
    f
end
plot_process()


T = 0.1
tt = 0.0:0.000001:T
obs = observation(T , ([100.0, -1.0] , ν) )

drift(M, t, a) = ∇logg(M, t, a, obs)
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
X = StochasticDevelopment(heun(), W, drift, (x₀,ν),M)
xx = map(x -> x[1], X.yy)

samplepaths = [xx]
for k in 1:10
    W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
    StochasticDevelopment!(heun(), X,  W, drift, (x₀, ν), M)
    push!(samplepaths, map(x -> x[1], X.yy))
end

CairoMakie.activate!()
function plot_processes()
    f = Figure(resolution=(5000, 3000), size = (5000,3000))
    ax = Axis(f[1,1], show_axis = false , show_grid = false)
    for k in eachindex(samplepaths)
        Makie.lines!(ax, map(x -> x[1], samplepaths[k]) , map(x -> x[2], samplepaths[k]), linewidth = 3.0, color = palette(:nuuk25)[end-k])
    end
    Makie.scatter!(ax,[ x₀[1]], [x₀[2]], color = palette(:nuuk25)[end-length(samplepaths)], markersize = 40)
    Makie.scatter!(ax,[obs.u[1][1]], [obs.u[1][2]], color = palette(:nuuk25)[end-length(samplepaths)], markersize = 40)
    hidespines!(ax) ; hidedecorations!(ax)
    f
end


fig_web = with_theme(plot_processes, theme_web())
save(outdir*"EuclideanBridges1.png", fig_web, px_per_unit=4)
Makie.save(outdir*"EuclideanBridges.png", fig)