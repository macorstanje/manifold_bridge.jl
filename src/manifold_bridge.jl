using Manifolds
using ForwardDiff
using Bridge
using StaticArrays
using LinearAlgebra
using Einsum
using LegendrePolynomials
using LaTeXStrings
using Plots
using GLMakie, CairoMakie, Makie



struct observation
    t::Real
    u::Tuple{Vector{Float64}, Matrix{Float64}}
end

function Frame(x)
    N = Array{Float64}(Manifolds.normal_vector(M,x))
    ν = nullspace(N')
    return (x, ν)
end

include("horizontal_development.jl")

include("manifold_plots.jl")

include("torus.jl")

include("inference.jl")