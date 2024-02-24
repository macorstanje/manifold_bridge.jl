include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")
GLMakie.activate!()
using Random
using Distributions


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

function christoffel_symbols_second(M::Ellipsoid, B::AbstractBasis, p)
    Γ = zeros(2,2,2)
    u,v = Manifolds.get_parameters(M, B.A, B.i, p)
    den = 1+u^2+v^2
    Γ[1,:,:] = (2/den) .* [-u -v ; -v u]
    Γ[2,:,:] = (2/den) .* [v -u ; -u -v]
    return Γ
end


tt = 0.0:0.001:100.0
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
γ = deepcopy(W) ; for i in eachindex(γ.tt) γ.yy[i] = 1.0.*[γ.tt[i], 2*γ.tt[i]] end
drift(M,B,t,a) = zeros(manifold_dimension(M))
X = StochasticDevelopment(heun(), W, drift, (p, ν), M, A)
xx = map(x -> x[1], X.yy)

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
xT = [0.0, 1.0,0.0]

g(M::Manifolds.Sphere, B::AbstractBasis, t, a, xT) = κ(T-t, a, Manifolds.get_parameters(M,B.A,B.i, xT), M, B)

function cometric(a, M, B::AbstractBasis)
    u,v = a[1], a[2]
    return 0.25*(1+u^2+v^2)*I
end

function ∇logg(M::Manifolds.Sphere, B::AbstractBasis, t, a, xT)
    _∇ = ForwardDiff.gradient(a -> log(g(M, B, t, a, xT)), a)
    g⁺ = cometric(a, M, B)
    return g⁺*_∇
end


T = 1.0
tt = 0.0:0.001:T
W = Bridge.sample(tt, Wiener{SVector{2, Float64}}())
drift(M,B,t,a) = ∇logg(M,B, t, a, xT)
X = StochasticDevelopment(heun(), W, drift, (p, ν), M, A)
xx = map(x -> x[1], X.yy)

ax, fig = sphere_figure()
lin = lines!(ax, map(x -> x[1], xx), map(x -> x[2], xx) , map(x -> x[3], xx) ; linewidth = 4.0, color = palette(:default)[1])
x0_pt  = Makie.scatter!(ax, p[1],p[2],p[3], color = :red, markersize = 25, label = L"$x_0$")
xT_pt = Makie.scatter!(ax, xT[1],xT[2],xT[3], color = :blue, markersize = 25, label = L"$x_T$")
axislegend(ax;
        labelsize = 50, 
        framewidth = 1.0, 
        orientation = :vertical,
        patchlabelgap = 18,
        patchsize = (50.0,50.0),
        margin = (320.0,320.0,320.0,320.0))
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

