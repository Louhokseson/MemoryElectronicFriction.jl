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
# CLI argument parsing — accepts --vib <int>, --temperature <int>,
# --translational_kinetic <float, eV>, --ntraj <int> and --stride <int>.
# --ntraj N runs only the first N trajectories of each MD file and adds
# `ntraj=N` to the output name, so a quick test never takes the full run's
# filename; --stride (default 1) is part of the name too. Keep stride·dt well
# below the ~18 fs NO vibration (stride 4 → 1 fs); stride ≳ 30 aliases it and
# ΔE is meaningless. Quick test:
#   --vib 16 --translational_kinetic 1.0 --temperature 300 --ntraj 2 --stride 4
# Falls back to full sweep when run interactively (no args).
# ---------------------------------------------------------------------------

let i = 1, _vib = nothing, _temp = nothing, _tk = nothing, _ntraj = nothing, _stride = nothing
    while i <= length(ARGS)
        if ARGS[i] == "--vib" && i < length(ARGS)
            _vib  = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--temperature" && i < length(ARGS)
            _temp = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--translational_kinetic" && i < length(ARGS)
            _tk   = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--ntraj" && i < length(ARGS)
            _ntraj = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--stride" && i < length(ARGS)
            _stride = parse(Int, ARGS[i+1]); i += 2
        else
            i += 1
        end
    end
    global const CLI_VIB    = _vib
    global const CLI_TEMP   = _temp
    global const CLI_TK     = _tk
    global const CLI_NTRAJ  = _ntraj
    global const CLI_STRIDE = _stride
end

# ---------------------------------------------------------------------------
# NOAu MD parameters. Must match run_md.jl exactly so dict_to_data_savename
# resolves to the same .h5 path the sweep wrote.
# ---------------------------------------------------------------------------

all_params_NOAu = Dict{String, Any}(
    "mass"                  => [[14.007 * 15.999 / (14.007 + 15.999), 14.007 + 15.999] .* u"u"],   # [μ_NO, M_NO]: r carries μ, z carries M
#    "Γ"                     => [1.5u"eV"], ## constant 1.5 eV
    "r0"                    => [[1.15u"Å", 5.0u"Å"]],    # (r, z); r0[1] is the frozen bond length when vibrational_state=nothing
    "translational_kinetic" => CLI_TK === nothing ? [4.0u"eV"] : [CLI_TK * u"eV"],
    "state"                 => [1],
    "tmax"                  => [1000.0u"fs"],
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
    "stride"         => CLI_STRIDE === nothing ? 1 : CLI_STRIDE,
    "parallel"       => nworkers() > 1,
    "kernel_average" => :arithmetic,                        # :arithmetic | :geometric
)
CLI_NTRAJ === nothing || (CPA_config_noau["ntraj"] = CLI_NTRAJ)

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
        partial_path   = full_data_path * ".partial"

        if isfile(full_data_path)
            @info "Skipping (already saved)" full_data_path
            continue
        end

        ads   = build_adsorbate(Val(model), p)
        Δt_au = austrip(p["dt"])
        # Per-trajectory ΔE is a length-D vector (per-DOF row decomposition),
        # NOAu → D=2 with rows [ΔE_r; ΔE_z]. --ntraj N keeps the first N
        # trajectories (column i ↔ MD trajectory_i).
        ntraj = min(get(CPA_config, "ntraj", length(trajs)), length(trajs))

        # Each finished trajectory goes into `<name>.h5.partial` as
        # trajectory_<i>, so a job killed by the walltime keeps its work and
        # rerunning the same command resumes. The final file (D × ntraj
        # `DeltaE_au`) is written only once all ntraj are done.
        done = isfile(partial_path) ? h5open(f -> Set(keys(f)), partial_path, "r") : Set{String}()
        todo = [i for i in 1:ntraj if "trajectory_$i" ∉ done]
        @info "Memory CPA" savingname n_todo=length(todo) n_done=ntraj - length(todo) partial_path nworkers=nworkers()

        t0 = time()
        for (n, i) in enumerate(todo)
            ΔE = delta_energy(model, ads, trajs[i], Δt_au;
                              ω_au=ω_au, T_au=T_au,
                              stride=stride, parallel=parallel,
                              kernel_average=kernel_average)
            h5open(f -> (f["trajectory_$i"] = ΔE), partial_path, "cw")

            # Full GC on the master and every worker after each trajectory. Each
            # Julia process sizes its heap against the whole node's memory, so 64
            # workers on one node let their garbage pile up until the node runs
            # out (ARCHER2 array 15625142: 8 tasks oom-killed after 9.7–15.4 h).
            # A full collection per trajectory keeps each heap near its live size;
            # max_worker_rss_GB (peak RSS of the largest worker) should stay flat.
            max_worker_rss_GB = if parallel
                GC.gc()
                @sync for w in workers()
                    @async remotecall_wait(GC.gc, w)
                end
                round(maximum(asyncmap(w -> remotecall_fetch(Sys.maxrss, w), workers())) / 2^30; digits = 2)
            else
                nothing
            end

            elapsed = time() - t0
            @info "ΔE done" trajectory=i progress="$(ntraj - length(todo) + n)/$ntraj" ΔE_eV=join(round.(ustrip.(ΔE .* auconvert(u"eV", 1)); sigdigits = 5), ", ") elapsed_min=round(elapsed / 60; digits = 1) eta_min=round(elapsed / n * (length(todo) - n) / 60; digits = 1) max_worker_rss_GB T_K stride
        end

        ΔE_au_mat = h5open(partial_path, "r") do f
            reduce(hcat, [read(f["trajectory_$i"]) for i in 1:ntraj])   # D × ntraj
        end
        h5open(full_data_path, "w") do fid
            fid["DeltaE_au"] = ΔE_au_mat
        end
        rm(partial_path)
        @info "Saved ΔE" full_data_path size=size(ΔE_au_mat)
    end
end

run_CPA_delta_energy(CPA_config_noau, params_list_NOAu, noau_trajs)