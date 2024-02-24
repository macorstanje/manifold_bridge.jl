
"""
    torus_figure()

This function generates a simple plot of a torus and returns the new figure containing the plot.
"""
function torus_figure(M::Manifolds.EmbeddedTorus)
    fig = Figure(resolution=(2000, 1600), size = (1200,1200), fontsize=46)
    ax = LScene(fig[1, 1], show_axis=false)
    ϴs, φs = LinRange(-π, π, 50), LinRange(-π, π, 50)
    param_points = [Manifolds._torus_param(M, θ, φ) for θ in ϴs, φ in φs]
    X1, Y1, Z1 = [[p[i] for p in param_points] for i in 1:3]
    gcs = [gaussian_curvature(M, p) for p in param_points]
    gcs_mm = max(abs(minimum(gcs)), abs(maximum(gcs)))
    pltobj = Makie.surface!(
        ax,
        X1,
        Y1,
        Z1;
        shading=true,
        ambient=Vec3f(0.65, 0.65, 0.65),
        backlight=1.0f0,
        color=gcs,
        colormap=Reverse(:RdBu),
        colorrange=(-gcs_mm, gcs_mm),
        transparency=true,
    )
    Makie.wireframe!(ax, X1, Y1, Z1; transparency=true, color=:gray, linewidth=0.5)
    # zoom!(ax.scene, cameracontrols(ax.scene), 0.98)
    #Colorbar(fig[1, 2], pltobj, height=Relative(0.5), label="Gaussian curvature")
    return ax, fig
end

function sphere_figure()
    fig = Figure(resolution=(2000, 1600), size = (1200,1200), fontsize=46)
    ax = LScene(fig[1, 1], show_axis=false)
    ϴs, φs = LinRange(0.0, 2π, 50), LinRange(0.0, π, 50)
    param_points = [[sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)] for θ in ϴs, φ in φs]
    X1, Y1, Z1 = [[p[i] for p in param_points] for i in 1:3]
    pltobj = Makie.surface!(
        ax,
        X1,
        Y1,
        Z1;
        # shading=true,
        ambient=Vec3f(0.65, 0.65, 0.65),
        backlight=1.0f0,
        colormap = :deep,
        colorrange = (10,100),
        transparency=true,
    )
    Makie.wireframe!(ax, X1, Y1, Z1; transparency=true, color=:gray, linewidth=0.5)
    # zoom!(ax.scene, cameracontrols(ax.scene), 0.98)
    #Colorbar(fig[1, 2], pltobj, height=Relative(0.5), label="Gaussian curvature")
    return ax, fig
end

struct Ellipsoid
    a::Float64
    b::Float64
    c::Float64
end

function ellipsoid_figure(M::Ellipsoid)
    fig = Figure(resolution=(2000, 1600), size = (1200,1200), fontsize=46)
    ax = LScene(fig[1, 1], show_axis=false)
    ϴs, φs = LinRange(0.0, 2π, 50), LinRange(0.0, π, 50)
    param_points = [[M.a*sin(θ)*cos(φ),M.b* sin(θ)*sin(φ), M.c*cos(θ)] for θ in ϴs, φ in φs]
    X1, Y1, Z1 = [[p[i] for p in param_points] for i in 1:3]
    pltobj = Makie.surface!(
        ax,
        X1,
        Y1,
        Z1;
        # shading=true,
        ambient=Vec3f(0.65, 0.65, 0.65),
        backlight=1.0f0,
        colormap = :deep,
        colorrange = (10,100),
        transparency=true,
    )
    Makie.wireframe!(ax, X1, Y1, Z1; transparency=true, color=:gray, linewidth=0.5)
    # zoom!(ax.scene, cameracontrols(ax.scene), 0.98)
    #Colorbar(fig[1, 2], pltobj, height=Relative(0.5), label="Gaussian curvature")
    return ax, fig
end