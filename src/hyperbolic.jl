"""
    Some basic manifold stuff implemented to hyperbolic spaces, which I could not get out of the manifolds.jl stuff
"""

function get_frame_parameterized(p, ν, M::Hyperbolic)
    _p, ν1, ν2 = HyperboloidPoint(p), HyperboloidTVector(ν[:,1]), HyperboloidTVector(ν[:,2])
    a = convert(PoincareBallPoint, _p).value
    Y1 = convert(PoincareBallTVector, _p ,ν1).value
    Y2 = convert(PoincareBallTVector, _p, ν2).value
    return a, hcat(Y1, Y2)
end

function get_frame_vectors(a, Y, M::Hyperbolic)
    _a, Y1, Y2 = PoincareBallPoint(a), PoincareBallTVector(Y[:,1]), PoincareBallTVector(Y[:,2])
    p = convert(HyperboloidPoint, _a).value
    ν1 = convert(HyperboloidTVector, _a, Y1).value
    ν2 = convert(HyperboloidTVector, _a, Y2).value
    return p, hcat(ν1, ν2)
end

import Manifolds.local_metric
function Manifolds.local_metric(M::Manifolds.Hyperbolic, a::Vector{Float64})
    return 4/(1-dot(a,a))^2 
end

Manifolds.local_metric(M::Hyperbolic, a::Manifolds.PoincareBallPoint) = local_metric(M, a.value)

import Manifolds.christoffel_symbols_second
function Manifolds.christoffel_symbols_second(M::Hyperbolic, a)
    d = manifold_dimension(M)
    den = 1-dot(a,a)
    Γ = zeros(d,d,d)
    Γ[1,:,:] = (2/den)*[a[1] a[2] ; a[2] -a[1]]
    Γ[2,:,:] = (2/den)*[-a[2] a[1] ; a[1] a[2]]
    return Γ
end
Manifolds.christoffel_symbols_second(M::Hyperbolic, a::PoincareBallPoint) = christoffel_symbols_second(M,a.value)
Manifolds.christoffel_symbols_second(M::Hyperbolic, a, B::AbstractBasis) = christoffel_symbols_second(M,a)


function logκ(t,x,y, M::Hyperbolic)
    ρ = distance(M,x,y)
    ζs = sqrt(t).*randn(5000)
    vars = [ζ > ρ ? ζ / sqrt(cosh(ζ) - cosh(ρ)) : 0.0 for ζ in ζs]
    dump(sum([z > ρ for z in ζs]))
    dump(ρ)
    return log(mean(vars))
end

function logg_param(M::Hyperbolic, t, a, obs::observation)
    logκ(obs.t-t, convert(HyperboloidPoint, PoincareBallPoint(a)).value , obs.u[1], M)
end

function ∇logg(M::Hyperbolic, t, a, obs::observation)
    inv(local_metric(M,a))*ForwardDiff.gradient(a -> logg_param(M, t, a, obs), a)
end


# Compute heat kernel when a grid is filled
"""
    gridρΔ

Mutable struct. Enter arrays of values for ρ = dist(x,xT) and Δ = T-t. Compute the 
value of logκ(ρ,Δ) at the grid values using fill_grid!(type::gridρΔ)
"""
mutable struct gridρΔ
    ρ
    Δ
    grid_vals
    gridρΔ(ρ,Δ) = new(ρ,Δ)
end

function fill_grid!(grid::gridρΔ, N, μ::Function)
    grid.grid_vals = zeros(length(grid.ρ), length(grid.Δ))
    p = Progress(length(grid.ρ)*length(grid.Δ))
    Z = randn(Int(N))
    for (i,ρ) in enumerate(grid.ρ)
        for (j,δ) in enumerate(grid.Δ)
            ζs = μ(ρ,δ) .+ sqrt(δ).*Z
            vars = [ζ > ρ ? ζ*exp(-μ(ρ,δ)*ζ/δ) / sqrt(cosh(ζ) - cosh(ρ)) : 0.0 for ζ in ζs]
            grid.grid_vals[i,j] = log(mean(vars))
            next!(p)
        end
    end
    grid
end

function linear_interpolation_2d(x,y,f,x₁,x₂,y₁,y₂)
    return [x₂-x, x-x₁]'*f*[y₂-y , y-y₁]/((x₂-x₁)*(y₂-y₁))
end

function linear_interpolation_1d(x,f,x₁,x₂)
    return ((x₂-x)*f[1] + (x-x₁)*f[2])/(x₁-x₂)
end

function logκ(t, x, y, grid::gridρΔ, M::Hyperbolic)
    ρ = distance(M,x,y)
    indρ = searchsortedfirst(grid.ρ, ρ)   
    indΔ = searchsortedfirst(grid.Δ, t)
    if indρ == length(grid.ρ) + 1
        if indΔ == length(grid.Δ) + 1
            return grid.grid_vals[end,end]
        else
            return linear_interpolation_1d(t, [grid.grid_vals[end,indΔ-1], grid.grid_vals[end,indΔ]], grid.Δ[indΔ-1], grid.Δ[indΔ])
        end
    else
        if indΔ == length(grid.Δ) + 1
            return linear_interpolation_1d(ρ, [grid.grid_vals[inddρ-1,end], grid.grid_vals[indρ,end]], grid.ρ[indρ-1], grid.ρ[indρ])
        else
            g = [grid.grid_vals[indρ-1, indΔ-1]  grid.grid_vals[indρ-1, indΔ] ; 
                    grid.grid_vals[indρ, indΔ-1] grid.grid_vals[indρ, indΔ]  ]
            return linear_interpolation_2d(ρ,t,g,grid.ρ[indρ-1],grid.ρ[indρ], grid.Δ[indΔ-1],grid.Δ[indΔ] )
        end
    end
end

δ(a,b) = 2*dot(a-b,a-b)/((1-dot(a,a))*(1-dot(b,b)))
dist(M::Hyperbolic,a,b) = acosh(1+δ(a,b))
∇dist(M,a,b) = inv(local_metric(M,a))*ForwardDiff.gradient(a -> dist(M,a,b), a)

function ∇logg(M::Hyperbolic, t, a, grid::gridρΔ, obs::observation)
    Δ = obs.t - t ; 
    x = convert(HyperboloidPoint, PoincareBallPoint(a)).value
    ρ = distance(M,x,obs.u[1])
    indΔ = searchsortedfirst(grid.Δ, t)
    indρ = searchsortedfirst(grid.ρ, t)
    if grid.grid_vals[indρ,indΔ] == -Inf
        return zeros(2)
    else
        if indρ > 1
            if grid.grid_vals[indρ-1,indΔ] == -Inf
                return zeros(2)
            else
                dρ = (grid.grid_vals[indρ, indΔ] - grid.grid_vals[indρ-1, indΔ])/(grid.ρ[indρ]-grid.ρ[indρ-1])
            end
        else
            return zeros(2)
        end
    end
    aT = convert(PoincareBallPoint, HyperboloidPoint(obs.u[1])).value
    return dρ*∇dist(M,a,aT)
end

# function logg_param(M::Hyperbolic, t, a, grid::gridρΔ, obs::observation)
#     logκ(obs.t-t, convert(HyperboloidPoint, PoincareBallPoint(a)).value , obs.u[1], grid, M)
# end

# function ∇logg(M::Hyperbolic, t, a, grid::gridρΔ, obs::observation)
#     inv(local_metric(M,a))*ForwardDiff.gradient(a -> logg_param(M, t, a, grid,obs), a)
# end




# function κ(t, y , z, M::Manifolds.Sphere, B::AbstractBasis)
#     yp = Manifolds.get_point(M, B.A, B.i, y)
#     zp = Manifolds.get_point(M, B.A, B.i, z)
#     sum([ exp(-k*(k+1)*t/2)*(2*k+1)/(2*pi)*LegendrePolynomials.Pl(dot(yp,zp),k) for k in 0:1:truncation ])
# end

# # Guiding function in local coordinates. 
# function g_param(M::Manifolds.Sphere, B::AbstractBasis, t, a, obs::observation) 
#     return κ(obs.t - t , a, get_parameters(M, B.A, B.i, obs.u[1] ), M, B)
# end


# # Riemannian gradient of log g, in local coordinates
# function ∇logg(M::Manifolds.Sphere, B::AbstractBasis, t, a, obs::observation)
#     _∇ = ForwardDiff.gradient(a -> log(g_param(M,B,t,a,obs)), a)
#     return Manifolds.local_metric(M, a, B) \ _∇
# end

# function κ(t, y , z, trunc, M::Manifolds.Sphere, B::AbstractBasis)
#     yp = Manifolds.get_point(M, B.A, B.i, y)
#     zp = Manifolds.get_point(M, B.A, B.i, z)
#     sum([ exp(-k*(k+1)*t/2)*(2*k+1)/(2*pi)*LegendrePolynomials.Pl(dot(yp,zp),k) for k in 0:1:trunc])
# end

# # Guiding function in local coordinates. 
# function g_param(M::Manifolds.Sphere, B::AbstractBasis, t, a, trunc, obs::observation) 
#     return κ(obs.t - t , a, get_parameters(M, B.A, B.i, obs.u[1] ), trunc, M, B)
# end


# # Riemannian gradient of log g, in local coordinates
# function ∇logg(M::Manifolds.Sphere, B::AbstractBasis, t, a, trunc, obs::observation)
#     _∇ = ForwardDiff.gradient(a -> log(g_param(M,B,t,a,trunc, obs)), a)
#     return Manifolds.local_metric(M, a, B) \ _∇
# end
# function Hor(i::Int64, u, M::Hyperbolic)
#     a, Y = u[1], u[2]
#     p = get_point3d(a,M)

# end

