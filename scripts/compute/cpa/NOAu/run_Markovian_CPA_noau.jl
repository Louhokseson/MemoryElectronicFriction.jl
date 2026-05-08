using Distributed
using DrWatson
@quickactivate "HokseonReproduce"

using NQCModels
using NQCDynamics
using NQCDynamics.InitialConditions: QuantisedDiatomic
using NQCCalculators
using LinearAlgebra: eigen
using Random
using Unitful, UnitfulAtomic
using HDF5
using HokseonAssistant
HokseonAssistant.julia_build_procs()

# Markovian friction is evaluated on master (sim.cache is mutated in place),
# so we only need the worker-side imports the memory `delta_energy` would
# otherwise require — keeping the @everywhere lines lets this script also be
# used to compare against the memory variant in the same session.
@everywhere using HokseonReproduce
@everywhere using StaticArrays: SA
@everywhere using Unitful, UnitfulAtomic

# ---------------------------------------------------------------------------
# Energy-loss engine. One level up — shared with the ET-only driver and the
# combined parent script.
# ---------------------------------------------------------------------------
include("../DeltaE.jl")

# ---------------------------------------------------------------------------
# NOAu MD parameters — must match run_md.jl exactly so dict_to_data_savename
# resolves to the same .h5 path the sweep wrote.
# ---------------------------------------------------------------------------

all_params_NOAu = Dict{String, Any}(
    "mass"                  => [(14.007 * 15.999 / (14.007 + 15.999)) * u"u"],
    "r0"                    => [[1.15u"Å", 5.0u"Å"]],
    "translational_kinetic" => [1.0u"eV"],
    "state"                 => [1],
    "tmax"                  => [500.0u"fs"],
    "dt"                    => [0.25u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [2],
    "termination_threshold" => [5.0u"Å"],
    "vibrational_state"     => [nothing],
    "trajectories"          => [1],
)
params_list_NOAu = dict_list(all_params_NOAu)

noau_trajs = [load_md_trajectories(p, "NOAu") for p in params_list_NOAu]

# ---------------------------------------------------------------------------
# Markovian-CPA configuration — no ω-grid, no kernel_average. The bath
# discretisation knobs (M, bw_eV) only feed the WideBandBath plumbing for
# `Simulation{DiabaticMDEF}`; the wide-band-exact friction is independent of
# them, so the defaults inside build_friction_sim are fine.
# ---------------------------------------------------------------------------

const CPA_config_noau = Dict{String, Any}(
    "model"   => :NOAu,
    "T_K"     => 300,
    "stride"  => 1,
    "parallel" => false,
)

# ---------------------------------------------------------------------------
# Run Markovian ΔE for one (CPA_config, params_list, trajs_list) triplet.
# ---------------------------------------------------------------------------

function run_Markovian_CPA_delta_energy(CPA_config, params_list, trajs_list)
    @unpack model, T_K, stride, parallel = CPA_config
    T_au = austrip(T_K * u"K")

    for (p, trajs) in zip(params_list, trajs_list)
        savingpath, savingname = CPA_dict_to_data_savename(p, CPA_config)
        full_data_path = datadir(savingpath, savingname)

        if isfile(full_data_path)
            @info "Skipping (already saved)" full_data_path
            continue
        end

        sim   = build_friction_sim(Val(model), p; T_au = T_au)
        Δt_au = austrip(p["dt"])
        ΔE_au_per_traj = Vector{Vector{Float64}}(undef, length(trajs))
        for (i, traj) in enumerate(trajs)
            ΔE_au_per_traj[i] = delta_energy(model, sim, traj, Δt_au;
                                        stride = stride, parallel = parallel)
            @info "Markovian ΔE done" model=model config=i stride T_K=CPA_config["T_K"] ΔE_eV=ustrip.(ΔE_au_per_traj[i] .* auconvert(u"eV", 1))
        end
        ΔE_au_mat = reduce(hcat, ΔE_au_per_traj)   # D × n_traj

        h5open(full_data_path, "w") do fid
            fid["DeltaE_au"] = ΔE_au_mat
        end
        @info "Saved Markovian ΔE" full_data_path size=size(ΔE_au_mat)
    end
end

run_Markovian_CPA_delta_energy(CPA_config_noau, params_list_NOAu, noau_trajs)