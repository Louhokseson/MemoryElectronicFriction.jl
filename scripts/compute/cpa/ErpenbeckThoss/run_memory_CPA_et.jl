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
# Energy-loss engine. One level up — shared with the NOAu-only driver and the
# combined parent script.
# ---------------------------------------------------------------------------
include("../DeltaE.jl")

# ---------------------------------------------------------------------------
# CLI argument parsing — accepts --gamma <float> and --temperature <int>.
# Falls back to full sweep when run interactively (no args).
# ---------------------------------------------------------------------------

let i = 1, _gamma = nothing, _temp = nothing
    while i <= length(ARGS)
        if ARGS[i] == "--gamma" && i < length(ARGS)
            _gamma = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--temperature" && i < length(ARGS)
            _temp  = parse(Int, ARGS[i+1]); i += 2
        else
            i += 1
        end
    end
    global const CLI_GAMMA = _gamma
    global const CLI_TEMP  = _temp
end

# ---------------------------------------------------------------------------
# ErpenbeckThoss MD parameters. Must match run_md.jl exactly so
# dict_to_data_savename resolves to the same .h5 path the sweep wrote.
# ---------------------------------------------------------------------------

all_params_et = Dict{String, Any}(
    "mass"                  => [10.54u"u"],
    "Γ"                     => CLI_GAMMA === nothing ? [0.02, 0.1, 0.25, 0.5, 1.0] .* u"eV" : [CLI_GAMMA * u"eV"],
    "r0"                    => [[5.0u"Å"]],              # 1 DOF: surface distance
    "translational_kinetic" => [0.5,1.0,2.0,3.0] .* u"eV",
    "state"                 => [1],
    "tmax"                  => [200.0u"fs"],
    "dt"                    => [0.01u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [1],                       # check r
    "termination_threshold" => [5.0u"Å"],                 # dissociation threshold
)
params_list_et = dict_list(all_params_et)

et_trajs = [load_md_trajectories(p, "ErpenbeckThoss") for p in params_list_et]

# ---------------------------------------------------------------------------
# DeltaE configuration.
# ---------------------------------------------------------------------------

const CPA_config_et = Dict{String, Any}(
    "model"          => :ErpenbeckThoss,
    "T_K"            => CLI_TEMP === nothing ? 2000 : CLI_TEMP,
    "ω"              => DEFAULT_ω_GRID_eV,         # see DeltaE.jl; eV, austrip'd inside
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
        # for HDF5 storage. ET → D=1.
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

run_CPA_delta_energy(CPA_config_et, params_list_et, et_trajs)