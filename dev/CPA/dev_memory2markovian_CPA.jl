# dev_memory2markovian_CPA.jl
#
# Sanity check: does the memory-CPA scheme reduce to Markovian CPA when the
# friction kernel Λ(ω; q) is replaced by its zero-frequency value Λ(0; q)?
#
# Reasoning. The memory-CPA energy loss is
#     ΔE_mem = ∫∫ V(t₁)ᵀ K(τ; q̄) V(t₂) dt₁ dt₂,    K(τ; q) = (2/π) ∫₀^∞ Λ(ω; q) cos(ωτ) dω
# If Λ(ω; q) is constant in ω at Λ₀(q) := Λ(ω→0; q), then the cosine
# transform gives K(τ; q) → Λ₀(q) δ(τ), so the double integral collapses
# to the Markovian single integral
#     ΔE_M = ∫ V(t)ᵀ Λ₀(q(t)) V(t) dt.
# Because Λ(ω→0) is exactly the wide-band-exact Markovian friction η(q) used
# by NQCD, the test ΔE here should agree with `dev_Markovian_CPA.jl`. Any
# residual gap is the truncation error from the finite ω-range we use to
# evaluate the cosine transform.
#
# Implementation. `delta_energy(...; zero_frequency=true)` (in DeltaE.jl)
# evaluates Λ once at ω_au[1] (the smallest ω in the grid — Lambda(ω) ∝ 1/ω
# so we can't pass exactly 0) and replicates it as a constant vector across
# the ω-grid before the cosine transform.
#
# Saving. The CPA config carries `"variant" => "test"` so results land at
#   data/sims/cpa/<model>/test/...
# disjoint from the genuine memory (`memory/`) and Markovian (`markovian/`)
# folders, leaving plot_Γ_DeltaE.jl untouched.

using Distributed
using DrWatson
@quickactivate "HokseonReproduce"

using NQCModels
using NQCDynamics
using NQCDynamics.InitialConditions: QuantisedDiatomic
using LinearAlgebra: eigen
using StaticArrays: SA
using Random
using Unitful, UnitfulAtomic
using HDF5
using HokseonAssistant
HokseonAssistant.julia_build_procs()

@everywhere using HokseonReproduce
@everywhere using StaticArrays: SA
@everywhere using Unitful, UnitfulAtomic

# DeltaE.jl: memory `delta_energy` with the new `zero_frequency` kwarg.
include("DeltaE.jl")

# ---------------------------------------------------------------------------
# MD parameters — must match run_md.jl exactly so the .h5 trajectory files
# resolve. Mirror dev_memory_CPA.jl / dev_Markovian_CPA.jl.
# ---------------------------------------------------------------------------

all_params_et = Dict{String, Any}(
    "mass"                  => [10.54u"u"],
    "Γ"                     => [0.02, 0.1, 0.25, 0.5, 1.0] .* u"eV",
    "r0"                    => [[5.0u"Å"]],
    "translational_kinetic" => [0.5,1.0,2.0,3.0] .* u"eV",
    "state"                 => [1],
    "tmax"                  => [200.0u"fs"],
    "dt"                    => [0.01u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [1],
    "termination_threshold" => [5.0u"Å"],
)
params_list_et = dict_list(all_params_et)

et_trajs = [load_md_trajectories(p, "ErpenbeckThoss") for p in params_list_et]

# ---------------------------------------------------------------------------
# CPA config — same shape as the memory config (so kernel_average is included
# in the savename hash) but with `"variant" => "test"` to redirect the output
# folder to sims/cpa/<model>/test/. The test still runs the memory machinery,
# only with Λ(ω) → Λ(0) collapsed inside `delta_energy(...; zero_frequency=true)`.
# ---------------------------------------------------------------------------

const CPA_config_et = Dict{String, Any}(
    "model"          => :ErpenbeckThoss,
    "T_K"            => 300,
    "ω"              => collect(0.01:0.01:20.0),
    "stride"         => 1,
    "parallel"       => nworkers() > 1,
    "kernel_average" => :arithmetic,
    "variant"        => "test",
)

# ---------------------------------------------------------------------------
# Run the zero-frequency-test ΔE over (CPA_config, params_list, trajs_list).
# Same loop structure as dev_memory_CPA.jl; only difference is
# `zero_frequency=true` and the explicit `"variant" => "test"` folder.
# ---------------------------------------------------------------------------

function run_memory2markovian_CPA_delta_energy(CPA_config, params_list, trajs_list)
    @unpack model, T_K, ω, stride, parallel, kernel_average = CPA_config

    T_au = austrip(T_K * u"K")
    ω_au = austrip.(ω * u"eV")

    for (p, trajs) in zip(params_list, trajs_list)
        savingpath, savingname = CPA_dict_to_data_savename(p, CPA_config)
        full_data_path = datadir(savingpath, savingname)

        if isfile(full_data_path)
            @info "Skipping (already saved)" full_data_path
            continue
        end

        ads   = build_adsorbate(Val(model), p)
        Δt_au = austrip(p["dt"])
        ΔE_au_vec = Vector{Float64}(undef, length(trajs))
        for (i, traj) in enumerate(trajs)
            ΔE_au_vec[i] = delta_energy(model, ads, traj, Δt_au;
                                 ω_au=ω_au, T_au=T_au,
                                 stride=stride, parallel=parallel,
                                 kernel_average=kernel_average,
                                 zero_frequency=true)
            @info "memory→Markovian test ΔE done" model=model config=i stride parallel kernel_average T_K=CPA_config["T_K"] ΔE_eV=ustrip(ΔE_au_vec[i] * auconvert(u"eV", 1))
        end

        h5open(full_data_path, "w") do fid
            fid["DeltaE_au"] = ΔE_au_vec
        end
        @info "Saved test ΔE" full_data_path n_traj=length(ΔE_au_vec)
    end
end

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------

run_memory2markovian_CPA_delta_energy(CPA_config_et, params_list_et, et_trajs)
