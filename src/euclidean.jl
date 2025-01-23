function ∇logg(M::Manifolds.Euclidean, t, x, obs::observation)
    (obs.u[1] - x)/(1*(obs.t - t))
end