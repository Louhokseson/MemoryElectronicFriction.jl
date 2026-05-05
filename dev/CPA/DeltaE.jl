# dev_DeltaE.jl — Memory-friction kernel ΔE computation
#
# Reusable engine for computing energy loss via the half-range Fourier cosine
# transform of the friction kernel Λ(ω; q).  Designed to be included from a
# driver script that has already set up Distributed workers.
#
# Usage (from a script that already called HokseonAssistant.julia_build_procs()):
#   include("dev_DeltaE.jl")
#   ads = build_adsorbate(Val(:NOAu), md_params)
#   ΔE  = delta_energy(:NOAu, ads, traj, dt_au; ω_au=..., T_au=...)

@everywhere using HokseonReproduce
@everywhere using StaticArrays: SA
@everywhere using Unitful, UnitfulAtomic
@everywhere using FFTW: bfft

# ---------------------------------------------------------------------------
# Helpers run on every worker: when parallel=true, the pmap closure invokes
# `kernel_at_position` (and through it `position_for_lambda` and
# `cosine_transform`) on each worker, so all three need to be defined there.
# ---------------------------------------------------------------------------
@everywhere begin
    # MD position vector (D-element column from traj.OutputPosition[:, i]) → the
    # position argument FrequencyLambda.Lambda expects.
    position_for_lambda(::Val{:ErpenbeckThoss}, q::AbstractVector) = q[1]
    position_for_lambda(::Val{:NOAu},           q::AbstractVector) = SA[q[1], q[2]]

    # Build the FrequencyLambda-compatible adsorbate model from the MD params.
    # ET: parameterised by Γ (matches NQCModels.ErpenbeckThoss in run_md.jl).
    # NOAu: parameterless — POGOModel and NOAuAdsorbate are the same physics.
    build_adsorbate(::Val{:ErpenbeckThoss}, md_params) =
        ErpenbeckThossAdsorbate(Γ = austrip(md_params["Γ"]))
    build_adsorbate(::Val{:NOAu}, _) = NOAuAdsorbate()

    # Half-range cosine transform via trapezoidal rule. ω, τ in atomic units;
    # Λ in atomic units. Same definition as plot_Lambda_NOAu_time.jl.


#=     function cosine_transform(ω_au::AbstractVector, Λ::AbstractVector,
                              τ_au::AbstractVector; pad::Int = 4)
        N_ω = length(ω_au)
        Δω  = ω_au[2] - ω_au[1]
        ω_0 = ω_au[1]
        @assert all(d -> isapprox(d, Δω; rtol = 1e-10), diff(ω_au)) "cosine_transform requires a uniform ω-grid"

        L = nextpow(2, pad * N_ω)

        h        = zeros(ComplexF64, L)
        h[1]     = 0.5 * Δω * Λ[1]
        @views h[2:N_ω-1] .= Δω .* Λ[2:N_ω-1]
        h[N_ω]   = 0.5 * Δω * Λ[N_ω]

        H̃ = bfft(h)                     # H̃[m+1] = Σ_k h[k] exp(+2πi m k / L)

        Δτ̃    = 2π / (L * Δω)
        τ̃_max = Δτ̃ * (L ÷ 2 - 1)

        out = Vector{Float64}(undef, length(τ_au))
        @inbounds for (n, τ) in enumerate(τ_au)
            if τ > τ̃_max
                out[n] = 0.0
                continue
            end
            m_real    = τ / Δτ̃
            m1        = floor(Int, m_real)
            frac      = m_real - m1
            H_interp  = (1 - frac) * H̃[m1 + 1] + frac * H̃[m1 + 2]
            out[n]    = (2/π) * real(cis(ω_0 * τ) * H_interp)
        end
        return out
    end =#

    function cosine_transform(ω_au::AbstractVector, Λ::AbstractVector,
                              τ_au::AbstractVector)
        Δω = diff(ω_au)
        map(τ_au) do τ
            integrand = Λ .* cos.(ω_au .* τ)
            (2/π) * sum(0.5 .* (integrand[1:end-1] .+ integrand[2:end]) .* Δω)
        end
    end 

    # K(q; τ_n)[k, l] for one trajectory position q. Returns D × D × Nτ.
    # `parallelism` is forwarded to FrequencyLambda.Lambda(::Vector, ...):
    #   :auto      — pmap if workers exist, else Threads.@threads (good on
    #                master).
    #   :threads   — force Threads.@threads. Use from inside a pmap worker
    #                so Lambda doesn't try to re-pmap (nested pmap).
    #   :serial    — plain loop. Use when an outer caller is already
    #                threading the ω axis and you want no nested parallelism.
    # 1DOF Λ comes back as Float64 / Vector{Float64}; NDOF as
    # Matrix{Float64} / Vector{Matrix{Float64}}.
    function kernel_at_position(model::Symbol, adsorbate, q::AbstractVector,
                                ω_au::AbstractVector, τ_au::AbstractVector,
                                T_au::Real; parallelism::Symbol = :auto)
        pos = position_for_lambda(Val(model), q)
        Λ_au = FrequencyLambda.Lambda(ω_au, adsorbate, pos, T_au;
                                      parallelism = parallelism)
        is_matrix = eltype(Λ_au) <: AbstractMatrix
        D = is_matrix ? size(first(Λ_au), 1) : 1
        K = zeros(D, D, length(τ_au))
        for k in 1:D, l in 1:D
            Λkl = is_matrix ? [m[k, l] for m in Λ_au] : Vector{Float64}(Λ_au)
            K[k, l, :] = cosine_transform(ω_au, Λkl, τ_au)
        end
        return K
    end
end

# ---------------------------------------------------------------------------
# Two-position kernel combiners. Add a new averaging scheme by defining one
# more `combine_kernel(::Val{:scheme}, a, b)` method; `delta_energy` will
# pick it up automatically via dispatch on `Val(kernel_average)`.
# ---------------------------------------------------------------------------
combine_kernel(::Val{:arithmetic}, a, b) = 0.5 * (a + b)
combine_kernel(::Val{:geometric},  a, b) = sqrt(a) * sqrt(b)
combine_kernel(::Val{S}, _, _) where {S} =
    throw(ArgumentError("unknown kernel_average=$S; define combine_kernel(::Val{:$S}, a, b)"))

# ---------------------------------------------------------------------------
# ΔE driver
# ---------------------------------------------------------------------------

"""
    delta_energy(model, adsorbate, traj, dt_au;
                 ω_au, T_au, stride=1, parallel=false,
                 kernel_average=:arithmetic) -> ΔE_au

Compute ΔE from the memory-kernel formula. `traj` is one element of the
loaded sweep (a NamedTuple with `:t`, `:OutputPosition`, `:OutputVelocity`).
`dt_au` is the *raw* MD step in atomic units; the effective `Δt` is
`stride·dt_au`. `model` is `:ErpenbeckThoss` or `:NOAu`. Returns ΔE in
atomic units (Hartree).

Set `parallel=true` to pmap the per-position kernel computation across
Distributed workers — recommended for `stride=1` on HPC. Workers must
already exist (`HokseonAssistant.julia_build_procs()` at the top of the
driver script creates them). Each worker computes Λ scalar-by-scalar over the
ω-grid, with `Threads.@threads` if it has more than one CPU.

`kernel_average` selects how the two-position kernel K(τ; q_i, q_j) is
approximated from the diagonal samples K(τ; q_i) and K(τ; q_j):
  * `:arithmetic` (default) → `0.5 * (K[i] + K[j])`
  * `:geometric`            → `sqrt(K[i]) * sqrt(K[j])`
"""
function delta_energy(model::Symbol, adsorbate, traj, dt_au::Real;
                      ω_au::AbstractVector, T_au::Real,
                      stride::Integer = 1, parallel::Bool = false,
                      progress::Bool = true,
                      kernel_average::Symbol = :arithmetic)
    combiner = Val(kernel_average)
    combine_kernel(combiner, 0.0, 0.0)   # fail-fast on unknown scheme
    idx  = 1:stride:length(traj.t)
    Q    = traj.OutputPosition[:, idx]
    V    = traj.OutputVelocity[:, idx]
    Δt   = stride * dt_au
    D, N = size(V)
    τ_au = (0:N-1) .* Δt

    K = if parallel
        progress && @info "Λ→K via pmap" N=N nworkers=nworkers()
        # Closure captures (model, adsorbate, ω_au, τ_au, T_au) — pmap
        # serialises them to each worker once per task. parallelism=:threads
        # tells FrequencyLambda to use the worker's local thread pool over
        # the ω-grid instead of re-pmap'ing (which would nest pmap).
        pmap(1:N) do i
            kernel_at_position(model, adsorbate, Q[:, i], ω_au, τ_au, T_au;
                               parallelism = :threads)
        end
    else
        Ks = Vector{Array{Float64,3}}(undef, N)
        for i in 1:N
            progress && @info "Λ→K at trajectory step" i=i of=N
            Ks[i] = kernel_at_position(model, adsorbate, Q[:, i], ω_au, τ_au, T_au;
                                        parallelism = :auto)
        end
        Ks
    end

    ΔE = 0.0
    @inbounds for i in 1:N, j in 1:i
        n = i - j + 1                        # τ_n = (i-j)·Δt → 1-based index
        for k in 1:D, l in 1:D
            Kkl = combine_kernel(combiner, K[i][k, l, n], K[j][k, l, n])
            ΔE += V[k, i] * Kkl * V[l, j]
        end
    end
    return ΔE * Δt^2
end
