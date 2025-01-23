include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")
using Random
using ProgressMeter
outdir = "/Users/marc/Documents/Onderzoek/ManifoldGPs/"
Random.seed!(61)

"""
    General forward simulation on the Torus
"""

M = Manifolds.EmbeddedTorus(3,1)
A = Manifolds.DefaultTorusAtlas()


# Initial point
x₀ = [3.0, 0.0, 1.0]
# Place in an induced basis 
i = Manifolds.get_chart_index(M, A, x₀)
a₀ = Manifolds.get_parameters(M, A, i, x₀)
i = (0.0, 0.0 )
B = induced_basis(M,A,i)
# Basis for tangent space to M at x₀
N = Array{Float64}(Manifolds.normal_vector(M,x₀))
ν = nullspace(N')

get_frame_parameterized(x₀, ν, M, B)
# Standard Brownian motion
tt = 0.0:0.001:1.0
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
drift(M,B,t,a) = zero(a)
X = StochasticDevelopment(heun(), W, drift, (x₀, ν) , M, A)
xx = map(x -> x[1], X.yy)

GLMakie.activate!()
fig = let
    ax, fig = torus_figure(M)
    lines!(ax, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 8.0, color = palette(:default)[1])
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
xT = [-3.0, 0.0,1.0]
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
                    linewidth = 4.0, color = palette(:oslo50)[5*k-1])
    end
    Makie.scatter!(ax, SVector{3, Float64}(x₀[1],x₀[2],x₀[3]), color = :red, markersize = 55, label = L"$x_0$")
    Makie.scatter!(ax, [xT[1],xT[2],xT[3]], color = :blue, markersize = 55, label = L"$x_T$")
    # axislegend(ax; 
    #         labelsize = 50, 
    #         framewidth = 1.0, 
    #         orientation = :vertical,
    #         patchlabelgap = 18,
    #         patchsize = (50.0,50.0),
    #         margin = (320.0,320.0,320.0,320.0))
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
θ₀ = [-4.0,4.0]
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
xT = X.yy[end][1]#[-3., 0., 2.]
obs = observation(T, Frame(xT))
drift(M,B,t,a) = V(θ₀, Φ)(a) .+ ∇logg(M,B,t,a,obs)
X = StochasticDevelopment(heun(), W, drift, (x₀, ν), M, A)
ll = loglikelihood(X, obs, θ₀, Φ, M, A)
llvals = [ll]


# Crank nicolson scheme
nr_simulations = 1000
xx = map(x -> x[1], X.yy)
samplepaths = [xx]
acc = [true]
Xᵒ = deepcopy(X)
prog = Progress(nr_simulations)
for k in 1:nr_simulations
    λ = .95
    Wᵒ = crank_nicolson(λ, W)
    StochasticDevelopment!(heun(), Xᵒ, Wᵒ, drift, (x₀, ν), M, A)
    llᵒ = loglikelihood(Xᵒ, obs, θ₀, Φ, M, A)
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

accepted_samplepaths = samplepaths[acc]

GLMakie.activate!()
fig = let
    nr_samplepaths_to_plot = acc[end]
    ax, fig = torus_figure(M)
    for k in 1:nr_samplepaths_to_plot
        lines!(ax, map(x -> x[1], samplepaths[end-k+1]), 
                    map(x -> x[2], samplepaths[end-k+1]) , 
                    map(x -> x[3], samplepaths[end-k+1]) ; 
                    linewidth = 4.0, color = palette(:oslo100)[max(1,Int64(floor(k*length(palette(:oslo100))/nr_samplepaths_to_plot)))])
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
Makie.save(outdir*"bridges_vector_field_0-1.png", fig)





"""
    Bridge simulation using Crank-Nicolson
"""
# Forward Trajectory
T = 1.0
tt = map((x) ->  x * (1.7 + (1-1.7)*x/T) , 0.0:0.001:T)
θ₀ = [-1, 0.1]
drift(M,B,t,a) = V(θ₀, Φ)(a)
W = sample(tt, Wiener{SVector{2, Float64}}())
X = StochasticDevelopment(heun(), W, drift,(x₀,ν), M, A)
xT = X.yy[end][1]

GLMakie.activate!()
fig = let
    ax, fig = torus_figure(M)
    lines!(ax, map(x -> x[1][1], X.yy), 
            map(x -> x[1][2], X.yy) , 
            map(x -> x[1][3], X.yy) ; 
            linewidth = 4.0, color = palette(:default)[1])
    Makie.scatter!(ax, Point3f0([x₀[1],x₀[2],x₀[3]]), color = :red, markersize = 25, label = L" $x_0$")
    Makie.scatter!(ax, Point3f0([xT[1], xT[2], xT[3]]), color = :blue, markersize = 25, label = L" $x_T$")
    axislegend(ax; 
            labelsize = 50, 
            framewidth = 1.0, 
            orientation = :vertical,
            patchlabelgap = 18,
            patchsize = (50.0,50.0),
            margin = (320.0,320.0,320.0,320.0))
    fig
end
Makie.save(outdir*"Forward_trajectory_-11.png", fig)

# Bridge to endpoint of forward observation
xT = [-3.,0.,2.]
obs = observation(T, Frame(xT))
drift(M,B,t,a) = V(θ₀, Φ)(a)  .+ ∇logg(M,B,t,a,obs)
W = sample(tt, Wiener{SVector{2, Float64}}())
X = StochasticDevelopment(heun(), W, drift, (x₀, ν), M, A)
Xᵒ = deepcopy(X)
ll = loglikelihood(X, obs, θ₀, Φ, M, A)
llvals = [ll]

# Runs of the Crank-Nicolson scheme
nr_simulations = 1000
xx = map(x -> x[1], X.yy)
samplepaths = [xx]
acc = [1]
prog = Progress(nr_simulations)
for k in 1:nr_simulations
    λ = 0.0
    Wᵒ = crank_nicolson(λ, W)
    StochasticDevelopment!(heun(), Xᵒ, Wᵒ, drift, (x₀, ν), M, A)
    llᵒ = loglikelihood(Xᵒ, obs, θ₀, Φ, M, A)
    llᵒ = isnan(llᵒ) ? -1e10 : llᵒ
    if isinf(llᵒ)
        break
    end
    if log(rand()) <= llᵒ - ll
        push!(acc, 1) # push!(acc, acc[end] + 1)
        W = Wᵒ
        X = Xᵒ
        ll = llᵒ
    else
        push!(acc, 0)
    end
    push!(llvals , ll)
    push!(samplepaths, map(x -> x[1], X.yy))
    next!(prog)
end
println("Terminated at $(length(acc)). Accepted percentage: $(100*mean(acc))%")
accepted_samplepaths = samplepaths[2:end][map(x -> x == 1.0, acc)]

# Plot of the behavior of the acceptance rate
CairoMakie.activate!()
fig = let 
    acc_percentage = [100*cumsum(acc)[i]/i for i in eachindex(acc)]
    fig = Figure(resolution=(2000, 1000), size = (1200,1200),fontsize=35)
    ax1 = Axis(fig[1, 1],xlabel = "Iteration", ylabel = "percentage accepted proposals")
    Makie.lines!(ax1, eachindex(acc), acc_percentage ; linewidth = 3.0, color = palette(:default)[1])
    fig
end

fig = let 
    fig = Figure(resolution=(2000, 1000), size = (1200,1200),fontsize=35)
    ax1 = Axis(fig[1, 1],xlabel = "Iteration", ylabel = "log-likelihood")
    Makie.lines!(ax1, eachindex(llvals), llvals ; linewidth = 3.0, color = palette(:default)[1])
    fig
end

GLMakie.activate!()
fig = let
    samplepaths_to_plot = length(samplepaths)-10:1:length(samplepaths)#map(i -> length(samplepaths)-i+1, 1:acc[end])
    # samplepaths_to_plot = 1:1:10 #vcat(1:1:10 , length(samplepaths)-10:1:length(samplepaths))
    ax, fig = torus_figure(M) 
    cols = length(palette(:oslo))
    for k in samplepaths_to_plot
        col = palette(:oslo)[max(1, Int64(ceil(cols - k*cols/samplepaths_to_plot[end])))]
        # col = palette(:oslo100)[max(1,length(palette(:oslo100)) - Int64(floor(k*length(palette(:oslo100))/length(samplepaths_to_plot))))]
        lines!(ax, map(x -> x[1], samplepaths[k]), 
                    map(x -> x[2], samplepaths[k]) , 
                    map(x -> x[3], samplepaths[k]) ; 
                    linewidth = 4.0, color = col)
    end
    Makie.scatter!(ax, x₀[1],x₀[2],x₀[3], color = :red, markersize = 25, label = L" $x_0$")
    Makie.scatter!(ax, xT[1],xT[2],xT[3], color = :blue, markersize = 25, label = L" $x_T$")
    axislegend(ax; 
            labelsize = 50, 
            framewidth = 1.0, 
            orientation = :vertical,
            patchlabelgap = 18,
            patchsize = (50.0,50.0),
            margin = (320.0,-160.0,320.0,320.0))
    GLMakie.Colorbar(fig[1,2], colormap = Reverse(:oslo), height= Relative(0.3),width = 60, 
            limits = (1, length(samplepaths)), label = "Iteration" )
    #         label = "Iteration", width = 30, alignmode  = Outside(10))
    fig
end
Makie.save(outdir*"Bridges-drift_CN_after_forward_trajectory_-11.png", fig)



struct Deformed_Torus
    R₁::Float64
    R₂::Float64
    r₁::Float64
    r₂::Float64
end


function deformed_torus_figure(M::Deformed_Torus)
    fig = Figure(resolution=(2000, 1600), size = (1200,1200), fontsize=46)
    ax = LScene(fig[1, 1], show_axis=false)
    ϴs, φs = LinRange(-π, π, 50), LinRange(-π, π, 50)
    param_points = [[(M.R₁+M.r₁*cos(θ))*cos(φ), (M.R₂+M.r₁*cos(θ))*sin(φ), M.r₂*sin(θ)]  for θ in ϴs, φ in φs]
    X1, Y1, Z1 = [[p[i] for p in param_points] for i in 1:3]
    gcs = [gaussian_curvature(Manifolds.EmbeddedTorus(R,a), p) for p in param_points]
    gcs_mm = max(abs(minimum(gcs)), abs(maximum(gcs)))
    pltobj = Makie.surface!(
        ax,
        X1,
        Y1,
        Z1;
        shading=true,
        ambient=Vec3f(0.65, 0.65, 0.65),
        backlight=1.0f0,
        color=gcs,
        colormap=Reverse(:RdBu),
        colorrange=(-gcs_mm, gcs_mm),
        transparency=true,
    )
    Makie.wireframe!(ax, X1, Y1, Z1; transparency=true, color=:gray, linewidth=0.5)
    # zoom!(ax.scene, cameracontrols(ax.scene), 0.98)
    #Colorbar(fig[1, 2], pltobj, height=Relative(0.5), label="Gaussian curvature")
    return ax, fig
end

function ϕ(M::Manifolds.EmbeddedTorus, N::Deformed_Torus, x)
    i = Manifolds.get_chart_index(M, A, x)
    θ, φ = Manifolds.get_parameters(M, A, i, x)
    sinθ, cosθ = sincos(θ + i[1])
    sinφ, cosφ = sincos(φ + i[2])
    return [(N.R₁ + N.r₁*cosθ)*cosφ , (N.R₂ + N.r₁*cosθ)*sinφ , N.r₂*sinθ]
end


N = Deformed_Torus(M.R, 5.0, 2.0, 1.0)
xx = map(x -> x[1], X.yy)
yy = map(x -> ϕ(M, N, x), xx)
y₀ = ϕ(M, N, x₀)

GLMakie.activate!()
fig = let
    ax, fig = deformed_torus_figure(N)
    lines!(ax, map(x -> x[1], yy), 
            map(x -> x[2], yy) , 
            map(x -> x[3], yy) ; 
            linewidth = 4.0, color = palette(:default)[1])
    Makie.scatter!(ax, y₀[1],y₀[2],y₀[3], color = :red, markersize = 25, label = L" $x_0$")
    Makie.scatter!(ax, yy[end][1], yy[end][2], yy[end][3], color = :blue, markersize = 25, label = L" $x_T$")
    axislegend(ax; 
            labelsize = 50, 
            framewidth = 1.0, 
            orientation = :vertical,
            patchlabelgap = 18,
            patchsize = (50.0,50.0),
            margin = (320.0,320.0,320.0,320.0))
    fig
end