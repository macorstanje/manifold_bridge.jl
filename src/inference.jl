"""
    Computes log g(0,x₀) + ∫ V(log g)(s, Xₛ) ds 
    for a SamplePath X, conditioned on hitting xT with a vector field V(θ, Φ)
"""

function loglikelihood(X::SamplePath, obs::observation , θ, Φ, M, A) 
    i = Manifolds.get_chart_index(M, A, X.yy[1][1])
    B = induced_basis(M,A,i)
    a = Manifolds.get_parameters(M, B.A, B.i, X.yy[1][1])

    out = log(g_param(M, B, X.tt[1] , a, obs)) # log g(0,x₀)
    for k in 1:length(X.tt)-1
        t, x = X.tt[k], X.yy[k][1] # tₖ, xₖ
        i = Manifolds.get_chart_index(M, A, x)
        B = induced_basis(M,A,i)
        a = Manifolds.get_parameters(M, A, i, x)

        _∇ = ForwardDiff.gradient(a -> log(g_param(M,B,t,a,obs)), a) 
        out += dot( V(θ, Φ)(a) , _∇ )*( X.tt[k+1] - t )
    end
    return out
end

function loglikelihood(X::SamplePath, obs::Array{observation, 1}, θ, Φ, M, A)
    times = map(o -> o.t, obs)
    
    i = Manifolds.get_chart_index(M, A, X.yy[1][1])
    B = induced_basis(M,A,i)
    a = Manifolds.get_parameters(M, B.A, B.i, X.yy[1][1])
    out = log(g_param(M, B, X.tt[1] , a, obs[getk(times, X.tt[1])] )) # log g(0,x₀)
    for k in 1:length(X.tt)-1
        t, x = X.tt[k], X.yy[k][1] # tₖ, xₖ
        i = Manifolds.get_chart_index(M, A, x)
        B = induced_basis(M,A,i)
        a = Manifolds.get_parameters(M, A, i, x)

        _∇ = ForwardDiff.gradient(a -> log(g_param(M,B,t,a,obs[getk(times, t)])), a) 
        out += dot( V(θ, Φ)(a) , _∇ )*( X.tt[k+1] - t )
    end
    return out
end

function getk(times::Array{T,1}, t::T) where {T<:Real}
    k::Int64 = searchsortedfirst(times, t)
    if times[k] == t 
        k += 1
    end
    return k
end

function loglikelihood(X::SamplePath, obs::observation , grid::gridρΔ, M::Hyperbolic) 
    a = convert(PoincareBallPoint, HyperboloidPoint(X.yy[1][1])).value

    out = logκ(X.tt[end], X.yy[1][1], obs.u[1], grid, M) # log g(0,x₀)
    for k in 1:length(X.tt)-1
        t, x = X.tt[k], X.yy[k][1] # tₖ, xₖ
        a = convert(PoincareBallPoint, HyperboloidPoint(x)).value

        _∇ = ∇logg(M, t, a, grid, obs)
        out += dot( V(a) , _∇ )*( X.tt[k+1] - t )
    end
    return out
end

function loglikelihood(X::SamplePath, obs::observation , V::Function, M::Hyperbolic, Zpos) 
    # a = convert(PoincareBallPoint, HyperboloidPoint(X.yy[1][1])).value
    out = 0.0
    #out = logκ(X.tt[end], X.yy[1][1], obs.u[1], grid, M) # log g(0,x₀)
    for k in 1:length(X.tt)-1
        t, x = X.tt[k], X.yy[k][1] # tₖ, xₖ
        a = convert(PoincareBallPoint, HyperboloidPoint(x)).value

        _∇ = ∇logg(M, t, a, obs, Zpos)
        out += dot( V(a) , _∇ )*( X.tt[k+1] - t )
    end
    return out
end

function loglikelihood(X::SamplePath, obs::Array{observation, 1}, V::Function, M::Hyperbolic, Zpos)
    times = map(o -> o.t, obs)
    # a = convert(PoincareBallPoint, HyperboloidPoint(X.yy[1][1])).value
    out = 0.0 # log(g_param(M, B, X.tt[1] , a, obs[getk(times, X.tt[1])] )) # log g(0,x₀)
    for k in 1:length(X.tt)-1
        t, x = X.tt[k], X.yy[k][1] # tₖ, xₖ
        a = convert(PoincareBallPoint, HyperboloidPoint(x)).value

        _∇ = ∇logg(M, t, a, obs[getk(times, t)], Zpos) 
        out += dot( V(a) , _∇ )*( X.tt[k+1] - t )
    end
    return out
end

function μΓ(M, A, X, Φ::Array{T,1}) where {T<:Function}
    μ = zeros(length(Φ))
    Γ = zeros(length(Φ), length(Φ))

    for j in 1:length(X.tt)-1
        x, ν = X.yy[j]
        i = Manifolds.get_chart_index(M, A, x)
        B = Manifolds.induced_basis(M, A, i)
        a, Y = get_frame_parameterized(x, ν, M, B)
        for k in eachindex(Φ)
            dx = Log(M, A, X.yy[j][1], X.yy[j+1][1])
            # μ[k] += dot( Y\Φ[k](a) , (W.yy[j+1] .- W.yy[j]) )
            # Riemannian inner product of Φ[k](a) and dx at a
            μ[k] += inner(M, B, a, Φ[k](a), dx)
            for ℓ in eachindex(Φ)
                Γ[k,ℓ] += inner(M, B, a, Φ[k](a), Φ[ℓ](a) )*( X.tt[j+1] - X.tt[j] )
                # Γ[k,ℓ] += dot(Y\Φ[k](a) , Y\Φ[ℓ](a))*(X.tt[j+1] - X.tt[j])
            end
        end
    end
    return μ, Γ
end

function μΓ(M::Hyperbolic, X, V::T) where {T<:Function}
    μ = 0.0 ; Γ = 0.0
    for j in 1:length(X.tt)-1
        x, ν = X.yy[j]
        a, Y = get_frame_parameterized(x, ν, M)
        dx = log(M, X.yy[j][1], X.yy[j+1][1])
        da = convert(PoincareBallTVector, HyperboloidPoint(x), HyperboloidTVector(dx)).value
        μ += local_inner_product(M, a, V(1)(a) , da)
        Γ += local_inner_product(M, a, V(1)(a), V(1)(a))*( X.tt[j+1] - X.tt[j] )
    end
    return μ, Γ
end

function crank_nicolson(η, W)
    W₂ = sample(W.tt, Wiener{SVector{length(W.yy[1]) , Float64}}())
    return SamplePath(W.tt, η .* W.yy + sqrt(1-η^2) .* W₂.yy)
end

function concat(BMs::Array{T,1}) where {T<:SamplePath}
    if length(BMs) == 1
        return BMs[1]
    end
    tt = vcat(BMs[1].tt[1:end-1], BMs[2].tt)
    yy = vcat(BMs[1].yy[1:end-1], BMs[2].yy)
    if length(BMs) == 2
        return SamplePath(tt,yy)
    end
    for i in 3:length(BMs)
        tt = vcat(tt[1:end-1], BMs[i].tt)
        yy = vcat(yy[1:end-1], BMs[i].yy)
    end
    return SamplePath(tt,yy)
end

separate(X::SamplePath, ind) = [ SamplePath(X.tt[ind[i]:ind[i+1]] , X.yy[ind[i]:ind[i+1]]) for i in 1:length(ind)-1]

