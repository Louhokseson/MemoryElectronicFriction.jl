# NegativeEigenvalue.jl — most negative eigenvalue of the 2×2 NO/Au Λ(ω; q).
#
# Engine shared by run_negative_eigenvalue_noau.jl (MD trajectories) and
# verify_negative_eigenvalue_fig3.jl (the Fig. 3 configurations), so the
# verification exercises exactly the production code. Defines the functions on
# every Distributed worker: include it after the workers exist.
#
# ω search: coarse scan on coarse_omega_grid(Δω, ω_max), then Brent inside the
# bracket [ω_{i-1}, ω_{i+1}] around the coarse minimum.

@everywhere using MemoryElectronicFriction
@everywhere using StaticArrays: SA, SMatrix
@everywhere using LinearAlgebra: Symmetric, eigen
@everywhere using Unitful, UnitfulAtomic
@everywhere import Optim

@everywhere begin
    # 2×2 Λ(ω; q, T) as a static symmetric matrix: analytic, allocation-free
    # eigen-decomposition, safe to call from many threads.
    friction_tensor(ω, ads, q, T) =
        Symmetric(SMatrix{2,2}(FrequencyLambda.Lambda(ω, ads, q, T)))

    eigmin_friction(ω, ads, q, T) = first(eigen(friction_tensor(ω, ads, q, T)).values)

    # min over ω of eigmin Λ(ω; q, T) → (λ_min, ω*, e_min). Coarse scan, then
    # Brent in the bracket around the coarse minimum. Brent is skipped when the
    # coarse minimum is non-negative (a = 0 there either way).
    function most_negative_eigenvalue(ads, q, T, ω_coarse, ω_tol)
        λ_coarse = [eigmin_friction(ω, ads, q, T) for ω in ω_coarse]
        i = argmin(λ_coarse)
        ω_star, λ_star = ω_coarse[i], λ_coarse[i]
        if λ_star < 0
            lo, hi = ω_coarse[max(i - 1, 1)], ω_coarse[min(i + 1, end)]
            res = Optim.optimize(ω -> eigmin_friction(ω, ads, q, T), lo, hi, Optim.Brent();
                                 abs_tol = ω_tol)
            if Optim.minimum(res) < λ_star
                ω_star, λ_star = Optim.minimizer(res), Optim.minimum(res)
            end
        end
        e = eigen(friction_tensor(ω_star, ads, q, T)).vectors[:, 1]
        e = e[2] < 0 ? -e : e        # quadratic forms are sign-blind; fix e_z ≥ 0
        return λ_star, ω_star, e
    end

    # One trajectory: t (N,), Q (2, N) in au. Threads over configurations.
    function negative_tail_along_trajectory(t, Q, ads, T, ω_coarse, ω_tol)
        N = length(t)
        λ_min  = zeros(N)
        ω_star = zeros(N)
        e_min  = zeros(2, N)
        Threads.@threads for n in 1:N
            q = SA[Q[1, n], Q[2, n]]
            λ_min[n], ω_star[n], e = most_negative_eigenvalue(ads, q, T, ω_coarse, ω_tol)
            e_min[:, n] .= e
        end
        return (; Time = t, OutputPosition = Q, a = min.(λ_min, 0.0),
                  lambda_min = λ_min, omega_star = ω_star, e_min)
    end
end

# Search defaults, shared so the verification runs the production settings.
const DEFAULT_DOMEGA_eV    = 0.25
const DEFAULT_OMEGA_TOL_eV = 1e-3

# MD Nyquist frequency ħπ/Δt in eV, rounded to 0.01 eV (8.27 eV at 0.25 fs).
nyquist_omega_max_eV(dt) = round(ustrip(auconvert(u"eV", π / austrip(dt))); digits = 2)

# Coarse ω grid Δω:Δω:ω_max in au, with ω_max appended when it is not a grid
# point, so the search window is exactly [Δω, ω_max].
function coarse_omega_grid(domega_eV, omega_max_eV)
    ω_grid = domega_eV:domega_eV:omega_max_eV
    ω_eV   = last(ω_grid) ≈ omega_max_eV ? collect(ω_grid) : [ω_grid; omega_max_eV]
    return austrip.(ω_eV .* u"eV")
end
