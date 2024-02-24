function get_frame_parameterized(p, ν, M, B)
    a = get_parameters(M, B.A, B.i, p)
    Y1 = zeros(manifold_dimension(M)) ; Y2 = zeros(manifold_dimension(M))
    Manifolds.get_coordinates_induced_basis!(M ,Y1 ,p, ν[:,1], B)
    Manifolds.get_coordinates_induced_basis!(M, Y2, p, ν[:,2], B)
    return a, hcat(Y1, Y2)
end

function get_frame_vectors(a, Y, M, B)
    p = get_point(M, B.A, B.i, a)
    ν1 = zeros(3) ; ν2 = zeros(3) ; ν3 = zeros(3)
    Manifolds.get_vector_induced_basis!(M, ν1, p, Y[:,1], B)
    Manifolds.get_vector_induced_basis!(M, ν2, p, Y[:,2], B)
    return p, hcat(ν1, ν2)
end

import Manifolds.inner
inner(M, B, a, X) = dot(X, Manifolds.local_metric(M,a, B)*X)
inner(M,B, a, X, Y) = dot(X, Manifolds.local_metric(M,a, B)*Y)



# In local coordinates
function Hor(i::Int64, u, M, B::AbstractBasis)
    # local coordinates of the point and the basis
    a, Y = u[1], u[2]
    # Get the point in the ambient space
    # p = get_point(M, B.A, B.i, a)
    # Christoffel symbols, for the torus implemented above
    Γ = Manifolds.christoffel_symbols_second(M, a, B)
    # Γ = christoffel_symbols_second(M, B, p)
    @einsum dY[k,m,i] :=  -Y[j,i]*Y[l,m]*Γ[k,j,l] # -Γ[i,k,l]*Y[k,m]*Y[l,j]
    return ( Y[:,i] ,  dY[:,:,i] )
end

function Hor(i::Int64, u, M::SymmetricPositiveDefinite, B::AbstractBasis)
    p, ν = u[1] , u[2]
    Γ = christoffel_symbols_second(M, B, p)
    E = get_basis(M,p,B).:data
    dp = ν[i]
    ζ = get_coordinates(M,p,ν[i],B)
    @einsum dζ[i,j,m] := -0.5*Γ[i,k,l]*ζ[k,m]*ζ[l,j]
    return (dp, get_vector(M,p, dζ[:,:,i], B))
end

abstract type SDESolver end
struct heun <: SDESolver end
struct euler <: SDESolver end

# Integrate stel in local coordinates (in basis B)
function IntegrateStep!(::heun, u, dZ, M, B::AbstractBasis)
    a, Y = u[1], u[2]

    # Euler forward step
    dā = sum([ Hor(i, (a,Y), M, B)[1].*dZ[i] for i in eachindex(dZ)])
    dȲ = sum([ Hor(i, (a,Y), M, B)[2].*dZ[i] for i in eachindex(dZ)])

    ā, Ȳ = a + dā , Y + dȲ
    
    # Stratonovich step using the euler next step 
    da = sum([0.5 .* (Hor(i, (ā,Ȳ), M, B)[1] .+ Hor(i, (a,Y), M, B)[1]).*dZ[i] for i in eachindex(dZ)])
    dY = sum([0.5 .* (Hor(i, (ā,Ȳ), M, B)[2] .+ Hor(i, (a,Y), M, B)[2]).*dZ[i] for i in eachindex(dZ)])

    a = a + da ; Y = Y + dY    
    u = (a,Y)
    u
end

function IntegrateStep!(::euler, u, dZ, M, B::AbstractBasis)
    a, Y = u[1], u[2]

    # Euler forward step
    da = sum([ Hor(i, (a,Y), M, B)[1].*dZ[i] for i in eachindex(dZ)])
    dY = sum([ Hor(i, (a,Y), M, B)[2].*dZ[i] for i in eachindex(dZ)])

    a = a + da ; Y = Y + dY    
    u = (a,Y)
    u
end


function IntegrateStep!(::heun, u, dZ, M::SymmetricPositiveDefinite, B::AbstractBasis)
    p,ν = u[1], u[2]

    dp̄ = sum([ Hor(i,(p,ν),M, B)[1].*dZ[i] for i in eachindex(dZ)])
    dν̄ = sum([ Hor(i,(p,ν),M, B)[2].*dZ[i] for i in eachindex(dZ)])

    println("p = $(p.:p)")
    println("dp̄ = $(dp̄)")
    p̄, ν̄ = Manifolds.exp(M,p,dp̄) , ν + dν̄ 
    
    dp = sum([0.5 .* (Hor(i,(p̄, ν̄),M, B)[1] .+ Hor(i,(p,ν),M,B)[1]).*dZ[i] for i in eachindex(dZ)])
    dν = sum([0.5 .* (Hor(i,(p̄, ν̄),M, B)[2] .+ Hor(i,(p,ν),M,B)[2]).*dZ[i] for i in eachindex(dZ)])

    p = Manifolds.exp(M,p,dp) ; ν = ν + dν  
    u = (p,ν)
    u
end


import Base.getindex, Base.setindex!
const .. = Val{:...}

setindex!(A::AbstractArray{T,1}, x, ::Type{Val{:...}}, n) where {T} = A[n] = x
setindex!(A::AbstractArray{T,2}, x, ::Type{Val{:...}}, n) where {T} = A[ :, n] = x
setindex!(A::AbstractArray{T,3}, x, ::Type{Val{:...}}, n) where {T} = A[ :, :, n] = x

getindex(A::AbstractArray{T,1}, ::Type{Val{:...}}, n) where {T} = A[n]
getindex(A::AbstractArray{T,2}, ::Type{Val{:...}}, n) where {T} = A[ :, n]
getindex(A::AbstractArray{T,3}, ::Type{Val{:...}}, n) where {T} = A[ :, :, n]

function Log(M::Manifolds.EmbeddedTorus, A, p, q)
    i = Manifolds.get_chart_index(M, A, p)
    B = induced_basis(M,A,i)
    θp, φp = Manifolds.get_parameters(M, A, i, p)
    θq, φq = Manifolds.get_parameters(M, A, i, q)
    return SVector{2, Float64}(rem2pi(θq - θp, RoundNearest) , rem2pi(φq - φp, RoundNearest))
end

# Overwrite Y
function StochasticAntiDevelopment!(Y::SamplePath, U::SamplePath, drift::Function, M, A::AbstractAtlas)
    N = length(U)
    N != length(Y) && error("U and Y differ in length.")
    tt = Y.tt
    yy = Y.yy
    uu = U.yy

    y = zeros(manifold_dimension(M))
    for k in 1:N-1
        yy[..,k] = y
        p, ν = uu[k][1], uu[k][2]
        i = Manifolds.get_chart_index(M, A, p)
        B = induced_basis(M,A,i)
        a, u = get_frame_parameterized(p,ν, M, B)
        dx = Log(M, A, uu[k][1], uu[k+1][1])
        # Manifolds.get_coordinates_induced_basis!(M, a, p, dx, B)
        dy = u \ ( dx - drift(M,B,tt[k],a)*(tt[k+1]-tt[k]))
        y += dy
        # Manifolds.get_coordinats_induced_basis!(M, z, p, dz, B)
        # println(y)
    end
    yy[..,N] = y
    Y
end

function StochasticAntiDevelopment(U::SamplePath, drift::Function, M, A::AbstractAtlas)
    let X = sample(U.tt, Wiener{SVector{manifold_dimension(M), Float64}}()); StochasticAntiDevelopment!(X, U, drift, M, A); X end
end

# Overwrite Y. 
function StochasticDevelopment!(::heun, Y::SamplePath, Z::SamplePath, drift::Function, u₀, M, A::AbstractAtlas)

    N = length(Y)
    N != length(Z) && error("Y and Z differ in length.")
    tt = Z.tt
    zz = Z.yy
    yy = Y.yy

    y::typeof(u₀) = u₀
    for k in 1:N-1
        yy[..,k]=y
        p, ν = y[1], y[2]

        # work within a chart
        i = Manifolds.get_chart_index(M, A, p)
        B = induced_basis(M,A,i)
        a, Y = get_frame_parameterized(p,ν, M, B)

        # euclidean process increment dZt such that (dUt = Hi(Ut)∘dZt); In case of BM, this is just dW
        dz = Y \ drift(M,B,tt[k],a)*(tt[k+1]-tt[k]) + zz[k+1] - zz[k]
        # update the local coordinates
        a, Y = IntegrateStep!(heun(), (a,Y), dz, M, B)
        # Map the result back to the ambient space
        y = get_frame_vectors(a,Y,M,B)
    end
    yy[..,N] = y
    Y
end



function StochasticDevelopment!(::euler, Y::SamplePath, Z::SamplePath, drift::Function, u₀, M, A::AbstractAtlas)

    N = length(Y)
    N != length(Z) && error("Y and Z differ in length.")
    tt = Z.tt
    zz = Z.yy
    yy = Y.yy

    y::typeof(u₀) = u₀
    for k in 1:N-1
        yy[..,k]=y
        p, ν = y[1], y[2]

        # work within a chart
        i = Manifolds.get_chart_index(M, A, p)
        B = induced_basis(M,A,i)
        a, Y = get_frame_parameterized(p,ν, M, B)

        # euclidean process increment dZt such that (dUt = Hi(Ut)∘dZt); In case of BM, this is just dW
        dz = Y \ drift(M,B,tt[k],a)*(tt[k+1]-tt[k]) + zz[k+1] - zz[k]
        # update the local coordinates
        a, Y = IntegrateStep!(euler(), (a,Y), dz, M, B)
        # Map the result back to the ambient space
        y = get_frame_vectors(a,Y,M,B)
    end
    yy[..,N] = y
    Y
end



function StochasticDevelopment(method::SDESolver, Z,drift, u₀, M, A)
    let X = Bridge.samplepath(Z.tt, deepcopy(u₀)); StochasticDevelopment!(method, X, Z, drift, u₀, M, A); X end
end

function HorizontalDevelopment!(method::SDESolver, Y::SamplePath, Z::SamplePath, u₀, M, A::AbstractAtlas)

    N = length(Y)
    N != length(Z) && error("Y and Z differ in length.")
    tt = Z.tt
    zz = Z.yy
    yy = Y.yy

    y::typeof(u₀) = u₀
    prog = Progress(N-1)
    for k in 1:N-1
        yy[..,k]=y
        p, ν = y[1], y[2]
        i = Manifolds.get_chart_index(M, A, p)
        B = induced_basis(M,A,i)
        a, Y = get_frame_parameterized(p,ν, M, B)
        dz = (Z.yy[k+1]-Z.yy[k])
        Hi = [ Hor(i, (a,Y), M, B) for i in eachindex(dz) ]
        da = sum([map(h -> h[1], Hi)[i].*dz[i] for i in eachindex(dz)])
        dY = sum([map(h -> h[2], Hi)[i].*dz[i] for i in eachindex(dz)])
        a += da ; Y += dY
        # a,Y = IntegrateStep!(method, (a,Y), dz, M, B)
        y = get_frame_vectors(a,Y,M,B)
    end
    yy[..,N] = y
    Y
end

function HorizontalDevelopment(method::SDESolver, Z, u₀, M, A)
    let X = Bridge.samplepath(Z.tt, deepcopy(u₀)); HorizontalDevelopment!(method, X, Z, u₀, M, A); X end
end