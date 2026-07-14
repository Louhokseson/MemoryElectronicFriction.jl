using Distributed
using DrWatson
@quickactivate "MemoryElectronicFriction"

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
@everywhere using MemoryElectronicFriction
@everywhere using StaticArrays: SA
@everywhere using Unitful, UnitfulAtomic

# ---------------------------------------------------------------------------
# Energy-loss engine. One level up — shared with the ET-only driver and the
# combined parent script.
# ---------------------------------------------------------------------------
include("../DeltaE.jl")

# ---------------------------------------------------------------------------
# CLI argument parsing — accepts --vib <int> and --temperature <int>.
# Falls back to full sweep when run interactively (no args).
# ---------------------------------------------------------------------------

let i = 1, _vib = nothing, _temp = nothing, _tk = nothing
    while i <= length(ARGS)
        if ARGS[i] == "--vib" && i < length(ARGS)
            _vib  = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--temperature" && i < length(ARGS)
            _temp = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--translational_kinetic" && i < length(ARGS)
            _tk   = parse(Float64, ARGS[i+1]); i += 2
        else
            i += 1
        end
    end
    global const CLI_VIB  = _vib
    global const CLI_TEMP = _temp
    global const CLI_TK   = _tk
end

# ---------------------------------------------------------------------------
# NOAu MD parameters. Must match run_md.jl exactly so dict_to_data_savename
# resolves to the same .h5 path the sweep wrote.
# ---------------------------------------------------------------------------

all_params_NOAu = Dict{String, Any}(
    "mass"                  => [(14.007 * 15.999 / (14.007 + 15.999)) * u"u"],   # μ_NO — POGO is 1-atom
#    "Γ"                     => [1.5u"eV"], ## constant 1.5 eV
    "r0"                    => [[1.15u"Å", 5.0u"Å"]],    # (r, z); r0[1] is the frozen bond length when vibrational_state=nothing
    "translational_kinetic" => CLI_TK === nothing ? [4.0u"eV"] : [CLI_TK * u"eV"],
    "state"                 => [1],
    "tmax"                  => [500.0u"fs"],
    "dt"                    => [0.25u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [2],                       # check z
    "termination_threshold" => [5.0u"Å"],                 # scattered threshold
    "vibrational_state"     => CLI_VIB  === nothing ? [0,3,16] : [CLI_VIB],
    "trajectories"          => [1000],
)
params_list_NOAu = dict_list(all_params_NOAu)

noau_trajs = [load_md_trajectories(p, "NOAu") for p in params_list_NOAu]

# ---------------------------------------------------------------------------
# DeltaE configuration.
# ---------------------------------------------------------------------------

const CPA_config_noau = Dict{String, Any}(
    "model"          => :NOAu,
    "T_K"            => CLI_TEMP === nothing ? 2000 : CLI_TEMP,
    "ω"              => DEFAULT_ω_GRID_eV,
    "stride"         => 1,
    "parallel"       => nworkers() > 1,
    "kernel_average" => :arithmetic,                        # :arithmetic | :geometric
)

# ---------------------------------------------------------------------------
# Run ΔE for one (CPA_config, params_list, trajs_list) triplet.
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
        # Per-trajectory ΔE is now a length-D vector (per-DOF row decomposition).
        # Collect into a Vector{Vector{Float64}} and hcat into a D × n_traj matrix
        # for HDF5 storage. NOAu → D=2 with rows [ΔE_r; ΔE_z].
        ΔE_au_per_traj = Vector{Vector{Float64}}(undef, length(trajs))
        for (i, traj) in enumerate(trajs)
            ΔE_au_per_traj[i] = delta_energy(model, ads, traj, Δt_au;
                                 ω_au=ω_au, T_au=T_au,
                                 stride=stride, parallel=parallel,
                                 kernel_average=kernel_average)
            @info "ΔE done" model=model config=i stride parallel kernel_average T_K=CPA_config["T_K"] ΔE_eV=ustrip.(ΔE_au_per_traj[i] .* auconvert(u"eV", 1))
        end
        ΔE_au_mat = reduce(hcat, ΔE_au_per_traj)   # D × n_traj

        h5open(full_data_path, "w") do fid
            fid["DeltaE_au"] = ΔE_au_mat
        end
        @info "Saved ΔE" full_data_path size=size(ΔE_au_mat)
    end
end

run_CPA_delta_energy(CPA_config_noau, params_list_NOAu, noau_trajs)