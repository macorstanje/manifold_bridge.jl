function TorusPlot(X::T, Y::T, Z::T, M::Manifolds.EmbeddedTorus) where {T<:AbstractArray}
    @assert Plots.backend() == Plots.PlotlyBackend() "Plotly() is not enabled"

    ϑ = LinRange(-pi, pi, 100)
    φ = LinRange(-pi, pi, 100)
    pts = [Manifolds._torus_param(M, θ, ϕ) for θ in ϑ, ϕ in φ]
    x, y, z = map(a -> a[1], pts), map(a -> a[2], pts), map(a -> a[3], pts)


    rng = M.R+M.r
    Plots.surface(x,y,z,
                    acis = true,
                    alpha= 0.5,
                    legend = false,
                    color = :grey,
                    xlim = (-rng-1, rng+1),
                    ylim = (-rng-1, rng+1),
                    zlim = (-M.r-1, M.r+1)
                    )
    
    Plots.plot!(X,Y,Z,
                    axis = true,
                    linewidth = 2.5,
                    color = palette(:default)[1],
                    legend = false,
                    xlabel = "x",
                    ylabel = "y",
                    zlabel = "z")
end


function TorusPlot(X::SamplePath{T},  M::Manifolds.EmbeddedTorus) where {T}
    X1 = extractcomp(X.yy,1)
    X2 = extractcomp(X.yy,2)
    X3 = extractcomp(X.yy,3)
    TorusPlot(X1, X2, X3, M)
end