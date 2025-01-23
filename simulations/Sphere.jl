include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")
GLMakie.activate!()
using Random
using Distributions
outdir = "/Users/marc/Documents/Onderzoek/ManifoldGPs/"


M = Manifolds.Sphere(2)
A = Manifolds.StereographicAtlas()

ax1, fig1 = sphere_figure()
# lines!(ax1, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 2.0, color = :green)
fig1
# Makie.save("sphere.png", fig1)

p = [0.0,0.0,1.0]
i = Manifolds.get_chart_index(M, A,p)
B = induced_basis(M,A,i)
N = p # normal vector
ν = nullspace(N')

a, Y = get_frame_parameterized(p,ν,M,B)

# function christoffel_symbols_second(M::Ellipsoid, B::AbstractBasis, p)
#     Γ = zeros(2,2,2)
#     u,v = Manifolds.get_parameters(M, B.A, B.i, p)
#     den = 1+u^2+v^2
#     Γ[1,:,:] = (2/den) .* [-u -v ; -v u]
#     Γ[2,:,:] = (2/den) .* [v -u ; -u -v]
#     return Γ
# end


tt = 0.0:0.0001:1.0
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
γ = deepcopy(W) ; for i in eachindex(γ.tt) γ.yy[i] = 1.0.*[γ.tt[i], 2*γ.tt[i]] end
drift(M,B,t,a) = zeros(manifold_dimension(M))
X = StochasticDevelopment(heun(), W, drift, (p, ν), M, A)
xx = map(x -> x[1], X.yy)

CairoMakie.activate!()
fig = let 
    fig = Figure( resolution=(2500, 2500), size = (1200,1200), fontsize=55)
    ax1 = Axis(fig[1, 1] , xlabel = L"$x$", ylabel = L"$y$")
    Makie.lines!(ax1, map(x -> x[1], W.yy), map(x -> x[2], W.yy), linewidth = 2.0, color = palette(:default)[1])
    Makie.scatter!(W.yy[1], color = :red, markersize = 40)
    fig
end
Makie.save(outdir*"BMR2.png", fig)

GLMakie.activate!()
fig = let
    ax1, fig1 = sphere_figure()
    lines!(ax1, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 2.0, color = palette(:default)[1])
    Makie.scatter!(SVector{3, Float64}(xx[1]), color = :red, markersize = 80)
    fig1
end
fig
Makie.save(outdir*"BMS2.png", fig)


Manifolds.log(M, X.yy[1][1], X.yy[2][1])

import ManifoldsBase.inner
inner(x) = dot(x,x)
function qu(xx)
    sum(inner.(diff(xx)))
end

qu(map(x -> x[1], X.yy))
qu(W.yy)

function QV_directional_frame(X)
    sum([ norm(Manifolds.log(M, X.yy[k][1], X.yy[k+1][1]))^2 for k in 1:length(X.tt)-1])
end


function wiener(a₀, W)
    X = [a₀]
    for i in eachindex(W.tt)[2:end]
        x = X[end]
        x = x + [0.0, 1/(2*tan(x[2]))].*((W.tt[i] - W.tt[i-1])) + [1/abs(sin(x[2])) 0.0 ; 0.0 1.0]*(W.yy[i]-W.yy[i-1])
        push!(X, x)
    end
    return SamplePath(W.tt, X)
end
params(a) = [cos(a[1])*sin(a[2]) , sin(a[1])*sin(a[2]), cos(a[2])]
a₀ = [pi/3,pi/3]
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
Y = wiener(a₀, W)

X = SamplePath(Y.tt, map(a -> params(a), Y.yy))
xx = X.yy
qu(xx)
QV_directional_frame(X)



function linear_combination(θ, ϕ::Array{T, 1}) where {T<:Function}
    K = typeof(θ) <: Real ? 1 : length(θ)
    return x -> K == 1 ? θ*ϕ[1](x) : sum([θ[k]*ϕ[k](x) for k in 1:K])
end
ϕ1(a) = [1., 0.]
ϕ2(a) = [0., 1.]
Φ = [ϕ1, ϕ2]
V(θ, Φ) = linear_combination(θ, Φ)


# Illustration of the vector field
θ₀ = [1.0,0.0]
Manifolds.get_chart_index(M, A, [1.,0.,0.])
fig = let
    N = 20
    θs, φs = LinRange(0.0, 2π, 2*N), LinRange(0.0, π, N)
    pts = [Point3f0([sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]) for θ in θs, φ in φs]
    pts_array = [reshape(pts, 2*N^2, 1)[i,1] for i in 1:2*N^2]
    vecs_array = [zeros(3) for i in 1:2*N^2]
    vecs = [reshape([V(θ₀./20, Φ)([θ, φ]) for θ in θs, φ in φs], 2*N^2, 1)[i,1] for i in 1:2*N^2]
    for j in eachindex(vecs_array)
        B = induced_basis(M, A, Manifolds.get_chart_index(M, A, pts[j]))
        Manifolds.get_vector_induced_basis!(M, vecs_array[j], pts_array[j], vecs[j], B)
        # vecs_array[j] = Point3f(vecs_array[j])
    end
    ax1, fig1 = sphere_figure()
    arrows!(ax1, pts_array, Point3f0.(vecs_array) ; arrowsize = Vec3f0(0.04,0.04,0.04), linewidth=0.02, color=palette(:default)[1])
    fig1
end
Makie.save(outdir*"vector_field_sphere.png", fig)


tt = 0.0:0.0001:1.0
θ₀ = [3.0,0.0]
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
drift(M,B,t,a) = V(θ₀, Φ)(a)
X = StochasticDevelopment(heun(), W, drift, (p, ν), M, A)
xx = map(x -> x[1], X.yy)

GLMakie.activate!()
fig = let
    ax1, fig1 = sphere_figure()
    lines!(ax1, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 2.0, color = palette(:default)[1])
    Makie.scatter!(SVector{3, Float64}(xx[1]), color = :red, markersize = 80)
    fig1
end
Makie.save(outdir*"vector_field_BM.png", fig)


N = 30
obs = xx[1:Int64(ceil(length(xx)/N)):length(xx)]
fig = let
    ax1, fig1 = sphere_figure()
    Makie.scatter!(SVector{3, Float64}(obs[1]), color = :blue, markersize = 80)
    for k in 2:N
        Makie.scatter!(SVector{3, Float64}(obs[k]), color = :red, markersize = 80)
    end
    fig1
end
Makie.save(outdir*"vector_field_points.png", fig)

σ = 0.2
noisy_obs = deepcopy(obs)
for (k,o) in enumerate(obs)
    noisy_obs[k] = exp(M, o, (I - o*o')*σ*randn(3))
end
fig = let
    ax1, fig1 = sphere_figure()
    Makie.scatter!(SVector{3, Float64}(obs[1]), color = :blue, markersize = 80)
    for k in 2:N
        Makie.scatter!(SVector{3, Float64}(noisy_obs[k]), color = :blue, markersize = 80)
    end
    fig1
end
Makie.save(outdir*"vector_field_points_noise.png", fig)

function Frame(x, M::Manifolds.Sphere)
    N = Array{Float64}(x)
    ν = nullspace(N')
    return (x, ν)
end

T = 1.0
xT = normalize(rand(3))
obs = observation(T, Frame(xT,M))
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
drift(M,B,t,a) = V(θ₀, Φ)(a) + ∇logg(M,B,t,a,obs)
X = StochasticDevelopment(heun(), W, drift, (p, ν), M, A)
xx = map(x -> x[1], X.yy)
fig = let 
    ax, fig1 = sphere_figure()
    lines!(ax, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 4.0, color = palette(:default)[1])
    Makie.scatter!(ax, SVector{3,Float64}(p[1],p[2],p[3]), color = :red, markersize = 80, label = L"$x_0$")
    Makie.scatter!(ax, SVector{3,Float64}(xT[1],xT[2],xT[3]), color = :blue, markersize = 80, label = L"$x_T$")
    # axislegend(ax;
    #     labelsize = 50, 
    #     framewidth = 1.0, 
    #     orientation = :vertical,
    #     patchlabelgap = 18,
    #     patchsize = (50.0,50.0),
    #     margin = (320.0,320.0,320.0,320.0))
    fig1
end
fig
Makie.save("spherebridge.png", fig)


## Transform to ellipsoid
M = Ellipsoid(4.5, 6.0, 3.0)

import Manifolds.get_chart_index
Manifolds.get_chart_index(M::Ellipsoid, A::Manifolds.StereographicAtlas, p) = Manifolds.get_chart_index(Manifolds.Sphere(2), A, p)
# function get_chart_index(::Ellipsoid, ::Manifolds.StereographicAtlas, p)
#     if p[1] < 0
#         return :south
#     else
#         return :north
#     end
# end

import Manifolds.get_parameters
function Manifolds.get_parameters!(M::Ellipsoid, x, ::Manifolds.StereographicAtlas, i::Symbol, p)
    p = inv(diagm([M.a, M.b, M.c]))*p
    if i === :north
        return x .= p[2:end] ./ (1 + p[1])
    else
        return x .= p[2:end] ./ (1 - p[1])
    end
end
function Manifolds.get_parameters(M, A, i, p)
    x = zeros(2)
    get_parameters!(M, x, A, i, p)
    x 
end


import Manifolds.get_point
function Manifolds.get_point!(M::Ellipsoid, p, ::Manifolds.StereographicAtlas, i::Symbol, x)
    xnorm2 = dot(x, x)
    dump(xnorm2)
    if i === :north
        p[1] = (1 - xnorm2) / (xnorm2 + 1)
    else
        p[1] = (xnorm2 - 1) / (xnorm2 + 1)
    end
    p[2:end] .= 2 * x / (xnorm2 + 1)
    return diagm([M.a, M.b, M.c])*p
end
function Manifolds.get_point(M::Ellipsoid,::Manifolds.StereographicAtlas, i::Symbol, x) 
    xnorm2 = dot(x, x)
    p = 2 .* x ./ (xnorm2 + 1)
    if i === :north
        pushfirst!(p,  (1 - xnorm2) / (xnorm2 + 1) )
    else
        pushfirst!(p, (xnorm2 - 1) / (xnorm2 + 1) )
    end
    return diagm([M.a, M.b, M.c])*p
end

import Manifolds.induced_basis
Manifolds.induced_basis(M::Ellipsoid,A,i) = Manifolds.induced_basis(Manifolds.Sphere(2), A, i)

import Manifolds.get_coordinates_induced_basis!
function Manifolds.get_coordinates_induced_basis!(
    M::Ellipsoid,
    Y::Vector{Float64},
    p::Vector{Float64},
    X::Vector{Float64},
    B::T,
) where {T<:InducedBasis}
    # p = diagm([M.a, M.b, M.c])*p
    n = 2
    if B.i === :north
        for i in 1:n
            Y[i] = X[i + 1] / (1 + p[1]) - X[1] * p[i + 1] / (1 + p[1])^2
        end
    else
        for i in 1:n
            Y[i] = X[i + 1] / (-1 + p[1]) - X[1] * p[i + 1] / (-1 + p[1])^2
        end
    end
    return Y
end

import  Manifolds.get_vector_induced_basis!
function Manifolds.get_vector_induced_basis!(
    M::Ellipsoid,
    Y,
    p,
    X,
    B::T,
) where {T<:InducedBasis}
    n = 2
    a = get_parameters(M, B.A, B.i, p)
    mult = inv(1 + dot(a, a))^2

    Y[1] = 0
    for j in 1:n
        Y[1] -= 4 * a[j] * mult * X[j]
    end
    for i in 2:(n + 1)
        Y[i] = 0
        for j in 1:n
            if i == j + 1
                Y[i] += 2 * (1 + dot(a, a) - 2 * a[i - 1]^2) * mult * X[j]
            else
                Y[i] -= 4 * a[i - 1] * a[j] * mult * X[j]
            end
        end
        if B.i === :south
            Y[i] *= -1
        end
    end
    return Y
end

ax, fig = ellipsoid_figure(M)
fig

# Map the bridge to the ellipsoid
S = Manifolds.Sphere(2)
φ(M::Ellipsoid, S::Manifolds.Sphere, x) = diagm([M.a,M.b,M.c])*x
φ⁻¹(M::Ellipsoid, S::Manifolds.Sphere, y) = inv(diagm([M.a,M.b,M.c]))*y
yy = map(x -> φ(M,S, x[1]), X.yy)

ax, fig = ellipsoid_figure(M)
lin = lines!(ax, map(x -> x[1], yy), map(x -> x[2], yy) , map(x -> x[3], yy) ; linewidth = 4.0, color = palette(:default)[1])
x0_pt  = Makie.scatter!(ax, φ(M,S,p)[1], φ(M,S,p)[2],φ(M,S,p)[3], color = :red, markersize = 25, label = L"$x_0$")
xT_pt = Makie.scatter!(ax, φ(M,S,xT)[1],φ(M,S,xT)[2],φ(M,S,xT)[3], color = :blue, markersize = 25, label = L"$x_T$")
axislegend(ax;
        labelsize = 50, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (320.0,320.0,320.0,320.0))
fig
Makie.save("mapped_bridge.png", fig)

# Simulate bridge on the ellipsoid
yT = φ(M::Ellipsoid, Manifolds.Sphere(2), xT)
function φ♯g(M::Ellipsoid, B, t, a, yT) 
    y = get_point(M, B.A, B.i, a)
    x = φ⁻¹(M, S, y)
    return g(S, B, t, Manifolds.get_parameters(Manifolds.Sphere(2), B.A, B.i, x), φ⁻¹(M, S, yT))
end

function cometric(a, M::Ellipsoid, B::AbstractBasis)
    cometric(a, S, B)
end

function φ♯∇logg(M::Ellipsoid, B::AbstractBasis, t, a, yT)
    _∇ = ForwardDiff.gradient(a -> log(φ♯g(M, B, t, a, yT)), a)
    g⁺ = cometric(a, M, B)
    return g⁺*_∇
end

T = 1.0
tt = 0.0:0.001:T
drift(M,B,t,a) = φ♯∇logg(M,B, t, a, yT)
Y = StochasticDevelopment(heun(), W, drift, (φ(M, Manifolds.Sphere(2), p), ν), M, A)
yy2 = map(x -> x[1], X.yy)




