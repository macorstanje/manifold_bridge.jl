using Manifolds
using ForwardDiff
#using Bridge
using StaticArrays
using LinearAlgebra
using Einsum
using LegendrePolynomials
using LaTeXStrings
using ProgressMeter
# using Plots
using GLMakie, CairoMakie, Makie#, ColorSchemes



struct observation
    t::Real
    u::Tuple{Vector{Float64}, Matrix{Float64}}
end

function Frame(x)
    N = Array{Float64}(Manifolds.normal_vector(M,x))
    ν = nullspace(N')
    return (x, ν)
end

"""
    Stuff from bridge.jl, wouldn't compile for some reason
"""

abstract type ContinuousTimeProcess{T} end
abstract type AbstractPath{T} end
struct SamplePath{T} <: AbstractPath{T}
    tt::Vector{Float64}
    yy::Vector{T}
    SamplePath{T}(tt, yy) where {T} = new(tt, yy)
end
SamplePath(tt, yy::Vector{T}) where {T} = SamplePath{T}(tt, yy)

import Base: copy, length
copy(X::SamplePath{T}) where {T} = SamplePath{T}(copy(X.tt), copy(X.yy))
length(X::SamplePath) = length(X.tt)


function samplepath(tt, v, ismut::Bool)
    ismut ? SamplePath(tt, [copy(v) for t in tt]) : SamplePath(tt, fill(v, length(tt)))
end
samplepath(tt, v) = samplepath(tt, v, ismutable(v))

struct VSamplePath{T} <: AbstractPath{T}
    tt::Vector{Float64}
    yy::Matrix{T}
    function VSamplePath(tt, yy::Matrix{T}) where {T}
        length(tt) != size(yy, 2) && throw(DimensionMismatch("length(tt) != size(yy, 2)"))
        new{T}(tt, yy)
    end
end

length(X::VSamplePath) = length(X.tt)


function endpoint!(X::SamplePath, v)
    X.yy[end] = v
    X
end

struct Wiener{T} <: ContinuousTimeProcess{T}
end
Wiener() = Wiener{Float64}()


function sample(tt, P::Wiener{T}) where T
    tt = collect(tt)
    yy = zeros(T,length(tt))
    sample!(SamplePath{T}(tt, yy), P)
end

function sample(tt, P::Wiener{T}, y1) where T
    tt = collect(tt)
    yy = zeros(T,length(tt))
    sample!(SamplePath{T}(tt, yy), P, y1)
end

mat(yy::Vector{SVector{d,T}}) where {d,T} = reshape(reinterpret(T, yy), d, length(yy))
function sample!(W::SamplePath{SVector{d,T}}, P::Wiener{SVector{d,T}}, y1 = zero(SVector{d,T})) where {d,T}
    sz = d
    W.yy[1] = y1
    yy = mat(W.yy)
    for i = 2:length(W.tt)
        rootdt = sqrt(W.tt[i]-W.tt[i-1])
        for j = 1:sz
            yy[sz*(i-1) + j] = yy[sz*(i-2) + j] + rootdt*randn(T)
        end
    end
    W
end

function sample!(W::VSamplePath{T}, P::Wiener{T}) where {T}
    N = length(W.tt)
    yy = W.yy
    sz = size(yy, 1)
    for i = 2:N
        rootdt = sqrt(W.tt[i] - W.tt[i-1])
        for j = 1:sz
            yy[j, i] = yy[j, i-1] + rootdt*randn(T)
        end
    end
    W
end

function sample!(W::SamplePath{T}, P::Wiener{T}, y1 = W.yy[1]) where T
    W.yy[1] = y1
    yy = W.yy
    for i = 2:length(W.tt)
        rootdt = sqrt(W.tt[i]-W.tt[i-1])
        yy[i] = yy[i-1] + rootdt*randn(T)
    end
    W
end


include("horizontal_development.jl")

include("manifold_plots.jl")

include("torus.jl")

include("sphere.jl")

include("hyperbolic.jl")

include("euclidean.jl")

include("inference.jl")