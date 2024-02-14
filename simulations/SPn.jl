include("/Users/marc/Documents/GitHub/manifold_bridge.jl/src/manifold_bridge.jl")



M = SymmetricPositiveDefinite(3)
AI = AffineInvariantMetric()
 EuclideanMetric()
p = SPDPoint(rand(M))
q = SPDPoint(rand(M))

log(M,p,q)

LE = Manifolds.LogEuclideanMetric()
M_LE = MetricManifold(M, LogEuclideanMetric())


B = DefaultOrthonormalBasis()
S = get_basis(M, Matrix{Float64}(I,3,3), B).:data

christoffel_symbols_second(M, p, B)

metric(MetricManifold(M,AI))

function G(M, p)
    bas = get_basis(M,p,DefaultOrthonormalBasis())
    d = manifold_dimension(M)
    return [Manifolds.inner(M, p, bas.:data[i], bas.:data[j]) for i in 1:d, j in 1:d]
end


function christoffel_symbols_second(M::SymmetricPositiveDefinite, B::AbstractBasis, p)
    d = manifold_dimension(M)
    return [dot(-(S[j]*S[k] + S[k]*S[j]), S[i])/2 for i in 1:d, j in 1:d, k in 1:d]
end

function StochasticDevelopment!(method::SDESolver, Y::SamplePath, Z::SamplePath, drift::Function, u₀, M::SymmetricPositiveDefinite, A::AffineInvariantMetric)
    N = length(Y)
    N != length(Z) && error("Y and Z differ in length.")
    tt = Z.tt
    zz = Z.yy
    yy = Y.yy

    y::typeof(u₀) = u₀
    for k in 1:N-1
        yy[..,k]=y
        p, ν = y[1], y[2]
        dz = zz[k+1] - zz[k] #inv(ν)*drift(M,B,tt[k],p)*(tt[k+1]-tt[k]) + zz[k+1] - zz[k]
        p,ν = IntegrateStep!(method, (p,ν), dz, M, B)
        y = (p,ν)
    end
    yy[..,N] = y
    Y
end

function StochasticDevelopment(method::SDESolver, Z,drift, u₀, M::SymmetricPositiveDefinite, A::AffineInvariantMetric)
    let X = Bridge.samplepath(Z.tt, deepcopy(u₀)); StochasticDevelopment!(method, X, Z, drift, u₀, M, A); X end
end

tt = 0.0:0.01:1.0
W = sample(tt, Wiener{SVector{6, Float64}}())
ν = get_basis(M,p,B).:data
StochasticDevelopment(heun(), W, (M,B,t,a) -> zeros(6), (p,ν), M, AI)

dp = Hor(1,(p,ν),M,B)[2]
Manifolds.exp(M,p,dp)
E = get_basis(M,p,B).:data