"""
    Some basic manifold stuff implemented to the embedded torus, which I could not get out of the manifolds.jl stuff
"""

# Riemannian Metric
import Manifolds.local_metric
function Manifolds.local_metric(M::Manifolds.EmbeddedTorus, a, B::T) where {T<:InducedBasis}
    # a = Manifolds.get_parameters(M, B.A, B.i, p)
    sinθ, cosθ = sincos(a[1] + B.i[1])
    return diagm([M.r^2, (M.R + M.r*cosθ)^2]) 
end

# Not used
import Manifolds.christoffel_symbols_second
function christoffel_symbols_second(M::Manifolds.EmbeddedTorus, B::Manifolds.AbstractBasis, p::T) where {T<:AbstractArray}
    θ, φ = Manifolds.get_parameters(M, B.A, B.i, p)
    θ += B.i[1] ; φ += B.i[2]
    sinθ, cosθ = sincos(θ)
    Γ = zeros(2,2,2)
    Γ[1,2,2] = (M.R + M.r*cosθ)*sinθ/M.r
    Γ[2,1,2] = -M.r*sinθ/(M.R + M.r*cosθ)
    Γ[2,2,1] = Γ[2,1,2]
    return Γ
end

const truncation = 10
function κ(t, y , z, M::Manifolds.EmbeddedTorus, B::AbstractBasis)
    θ1 = y[1] + B.i[1] ; ϕ1 = y[2] + B.i[2]
    θ2 = z[1] + B.i[1] ; ϕ2 = z[2] + B.i[2]
    out = sum([ exp(-(θ1 - θ2 - 2*pi*k)^2/(4*t/M.r^2) - (ϕ1 - ϕ2 - 2*pi*l)^2/(4*t/M.R^2)) 
                for k in -truncation:truncation, l in -truncation:truncation])
    return out/(4*pi*t/(M.R*M.r))
end

# Guiding function in local coordinates. 
function g_param(M::Manifolds.EmbeddedTorus, B::AbstractBasis, t, a, obs::observation) 
    return κ(obs.t - t , a, get_parameters(M, B.A, B.i, obs.u[1] ), M, B)
end


# Riemannian gradient of log g, in local coordinates
function ∇logg(M::Manifolds.EmbeddedTorus, B::AbstractBasis, t, a, obs::observation)
    _∇ = ForwardDiff.gradient(a -> log(g_param(M,B,t,a,obs)), a)
    return Manifolds.local_metric(M, a, B) \ _∇
end

