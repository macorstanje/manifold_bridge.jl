

struct frame
    p
    ν
    M::AbstractManifold
end

Π(u::frame) = u.p

struct FrameBundle <: AbstractManifold 
    manifold::AbstractManifold
end
getM(FM::FrameBundle) = FM.manifold


import ManifoldsBase: manifold_dimension
manifold_dimension(FM::FrameBundle) = FM.manifold.dimension+FM.manifold.dimension^2

import ManifoldsBase: check_point
function check_point(FM::FrameBundle, u::frame; kwargs...)
    check_point(getM(FM), Π(u); kwargs...)
    for i in 1:manifold_dimension(getM(FM))
        check_vector(getM(FM), Π(u), ν[i,:])
    end
end


function Hor(i::Int64, u::frame, A::AbstractAtlas)
    p, ν, M = u.p, u.ν, u.M
    chart =  Manifolds.get_chart_index(M, A, p)
    Γ = christoffel_symbols_second(M, A, p, chart)

    if manifold_dimension(M) == 1
        dν = -ν^2*Γ
        return Tuple(ν, dν)
    else
        @einsum dν[i,j,m] = -0.5*Γ[i,k,l]*ν[k,m]*ν[l,j]
        return Tuple(ν[:,i], dν[:,:,i])
    end
end

abstract type SDESolver end

struct Heun <: SDESolver end

function IntegrateStep!(::Heun, u::frame, dZ, A::AbstractAtlas)
    p, ν, M = u.p, u.ν, u.M

    ū = deepcopy(u)
    dū = sum([Hor(i,u,A)*dZ[i] for i in eachindex(dZ)])
    ū.p, ū.ν = ū.p + dū[1] , ū.ν + dū[2]
    
    du = sum([0.5*(Hor(i,ū,A) + Hor(i,u,A))*dZ[i] for i in eachindez(dZ)])
    u.p, u.ν = u.p + du[1], u.ν + du[2]
    return u
end

function StochasticDevelopment!(method::SDESolver, Y::SamplePath, Z::SamplePath, u₀::frame, A::AbstractAtlas)
    M = u₀.M

    N = length(Y)
    N != length(Z) && error("Y and Z differ in length.")
    tt = Z.tt
    zz = Z.yy
    yy = Y.yy

    y::typeof(u₀) = u₀
    for k in 1:N-1
        yy[..,k]=y
        dz = zz[k+1] - zz[k]
        x = y.p
        y_temp = IntegrateStep!(method, y, dz, A)
    end
    yy[..,N] = y
    Y
end

function StochasticDevelopment(method::SDESolver, Z, u₀, A)
    let X = Bridge.samplepath(Z.tt, zero(u₀)); StochasticDevelopment!(method, X, Z, u₀), A; X end
end
