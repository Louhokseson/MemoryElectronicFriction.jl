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

# `julia_build_procs()` spawns workers; propagate the imports they need so the
# pmap path inside `delta_energy(...; parallel=true)` can call FrequencyLambda
# and our helpers on every worker.
@everywhere using HokseonReproduce
@everywhere using StaticArrays: SA
@everywhere using Unitful, UnitfulAtomic

# ---------------------------------------------------------------------------
# Energy-loss engine (helpers + delta_energy).  Kept in a separate file so
# other driver scripts can reuse it without copying code.
# ---------------------------------------------------------------------------
include("DeltaE.jl")

# ---------------------------------------------------------------------------
# Parameters — must match run_md.jl exactly so dict_to_data_savename resolves
# to the same .h5 path that the sweep wrote.
# ---------------------------------------------------------------------------

all_params_et = Dict{String, Any}(
    "mass"                  => [10.54u"u"],
    "Γ"                     => [0.25u"eV"],
    "r0"                    => [[5.0u"Å"]],              # 1 DOF: surface distance
    "translational_kinetic" => [3.0u"eV"],
    "state"                 => [1],
    "tmax"                  => [200.0u"fs"],
    "dt"                    => [0.01u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [1],                       # check r
    "termination_threshold" => [5.0u"Å"],                 # dissociation threshold
)
params_list_et = dict_list(all_params_et)

all_params_NOAu = Dict{String, Any}(
    "mass"                  => [(14.007 * 15.999 / (14.007 + 15.999)) * u"u"],   # μ_NO — POGO is 1-atom
#    "Γ"                     => [1.5u"eV"], ## constant 1.5 eV
    "r0"                    => [[1.15u"Å", 5.0u"Å"]],    # (r, z); r0[1] is the frozen bond length when vibrational_state=nothing
    "translational_kinetic" => [1.0u"eV"],
    "state"                 => [1],
    "tmax"                  => [500.0u"fs"],
    "dt"                    => [0.25u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [2],                       # check z
    "termination_threshold" => [5.0u"Å"],                 # scattered threshold
    # nothing → frozen bond (old behaviour). Integer ν → EBK-sample (r, ṙ)
    # at quantum number ν; bump trajectories to ~1000 for a ν ensemble.
    "vibrational_state"     => [nothing],                 # try [nothing, 0, 3, 16]
    "trajectories"          => [1],
)
params_list_NOAu = dict_list(all_params_NOAu)

# ---------------------------------------------------------------------------
# Load trajectories. `load_md_trajectories(params, model_folder; outputs=...)`
# returns a Vector of NamedTuples, one per trajectory in the .h5 file. Each
# NamedTuple is keyed by `:t` plus whatever outputs were requested:
#   .t              :: Vector{Float64}   (Nt,)      time, atomic units
#   .OutputPosition :: Matrix{Float64}   (D × Nt)   D=1 for ET, D=2 for NOAu
#   .OutputVelocity :: Matrix{Float64}   (D × Nt)
# `traj.OutputPosition[1, :]` is the bond coordinate (both models);
# `traj.OutputPosition[2, :]` is z (NOAu only). Trajectory lengths Nt vary
# because of the TerminatingCallback — hence a Vector, not a rectangular array.
#
# Default `outputs` is (:OutputPosition, :OutputVelocity). Pass a different
# tuple to read other fields written by FileReduction, e.g.
#   load_md_trajectories(p, "NOAu";
#       outputs = (:OutputPosition, :OutputKineticEnergy, :OutputTotalEnergy))
# Scalar outputs (energies) come back as Vector{Float64} of length Nt.
# ---------------------------------------------------------------------------

et_trajs   = [load_md_trajectories(p, "ErpenbeckThoss") for p in params_list_et]
noau_trajs = [load_md_trajectories(p, "NOAu")           for p in params_list_NOAu]

# ---------------------------------------------------------------------------
# DeltaE configuration — one dict per model, independent from MD params.
# ---------------------------------------------------------------------------

const CPA_config_et = Dict{String, Any}(
    "model"          => :ErpenbeckThoss,
    "T_K"            => 300,
    "ω"              => DEFAULT_ω_GRID_eV,         # see DeltaE.jl; eV, austrip'd inside
    "stride"         => 100,
    "parallel"       => nworkers() > 1,
    "kernel_average" => :arithmetic,                        # :arithmetic | :geometric
)

const CPA_config_noau = Dict{String, Any}(
    "model"          => :NOAu,
    "T_K"            => 300,
    "ω"              => DEFAULT_ω_GRID_eV,
    "stride"         => 1,
    "parallel"       => nworkers() > 1,
    "kernel_average" => :arithmetic,                        # :arithmetic | :geometric
)

# ---------------------------------------------------------------------------
# Run ΔE for a single (CPA_config, params_list, trajs_list) triplet.
# ---------------------------------------------------------------------------

function run_CPA_delta_energy(CPA_config, params_list, trajs_list)

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
                                 kernel_average=kernel_average)
            @info "ΔE done" model=model config=i stride parallel kernel_average T_K=CPA_config["T_K"] ΔE_eV=ustrip(ΔE_au_vec[i] * auconvert(u"eV", 1))
        end

        # Save ΔE_au_vec as a 1-D dataset in HDF5.
        h5open(full_data_path, "w") do fid
            fid["DeltaE_au"] = ΔE_au_vec
        end
        @info "Saved ΔE" full_data_path n_traj=length(ΔE_au_vec)
    end
end

# ---------------------------------------------------------------------------
# Line up ET and NOAu
# ---------------------------------------------------------------------------

run_CPA_delta_energy(CPA_config_et,   params_list_et,   et_trajs)
#run_CPA_delta_energy(CPA_config_noau, params_list_NOAu, noau_trajs)
