"""
    Some basic manifold stuff implemented to the embedded torus, which I could not get out of the manifolds.jl stuff
"""

# Riemannian Metric

import Manifolds.christoffel_symbols_second
function christoffel_symbols_second(M::Manifolds.Sphere, a, B::Manifolds.AbstractBasis )
    u, v = a[1], a[2]
    den = 1+u^2+v^2
    Γ = zeros(2, 2, 2)
    Γ[1,:,:] = (2/den) .* [-u -v ; -v u]
    Γ[2,:,:] = (2/den) .* [v -u ; -u -v]
    return Γ
end

const truncation = 10
function κ(t, y , z, M::Manifolds.Sphere, B::AbstractBasis)
    yp = Manifolds.get_point(M, B.A, B.i, y)
    zp = Manifolds.get_point(M, B.A, B.i, z)
    sum([ exp(-k*(k+1)*t/2)*(2*k+1)/(2*pi)*LegendrePolynomials.Pl(dot(yp,zp),k) for k in 0:1:truncation ])
end

# Guiding function in local coordinates. 
function g_param(M::Manifolds.Sphere, B::AbstractBasis, t, a, obs::observation) 
    return κ(obs.t - t , a, get_parameters(M, B.A, B.i, obs.u[1] ), M, B)
end


# Riemannian gradient of log g, in local coordinates
function ∇logg(M::Manifolds.Sphere, B::AbstractBasis, t, a, obs::observation)
    _∇ = ForwardDiff.gradient(a -> log(g_param(M,B,t,a,obs)), a)
    return Manifolds.local_metric(M, a, B) \ _∇
end

function κ(t, y , z, trunc, M::Manifolds.Sphere, B::AbstractBasis)
    yp = Manifolds.get_point(M, B.A, B.i, y)
    zp = Manifolds.get_point(M, B.A, B.i, z)
    sum([ exp(-k*(k+1)*t/2)*(2*k+1)/(2*pi)*LegendrePolynomials.Pl(dot(yp,zp),k) for k in 0:1:trunc])
end

# Guiding function in local coordinates. 
function g_param(M::Manifolds.Sphere, B::AbstractBasis, t, a, trunc, obs::observation) 
    return κ(obs.t - t , a, get_parameters(M, B.A, B.i, obs.u[1] ), trunc, M, B)
end


# Riemannian gradient of log g, in local coordinates
function ∇logg(M::Manifolds.Sphere, B::AbstractBasis, t, a, trunc, obs::observation)
    _∇ = ForwardDiff.gradient(a -> log(g_param(M,B,t,a,trunc, obs)), a)
    return Manifolds.local_metric(M, a, B) \ _∇
end