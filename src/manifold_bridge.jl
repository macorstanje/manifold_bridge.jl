module manifold_bridge

    using Plots
    using ForwardDiff
    using LinearAlgebra
    using StaticArrays
    using Einsum
    using Bridge
    using Manifolds

    export TorusPlot

    export frame, FrameBundle, Hor, SDESolver, Heun, IntegrateStep!
    export StochasticDevelopment!, StochasticDevelopment


    include("basics.jl")

    include("manifold_plots.jl")

    include("horizontal_development.jl")

end
