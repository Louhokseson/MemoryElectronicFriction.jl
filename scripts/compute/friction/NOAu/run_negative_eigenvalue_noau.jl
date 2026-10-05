# run_negative_eigenvalue_noau.jl — most negative eigenvalue of Λ(ω; q) along
# the NO/Au classical MD trajectories.
#
# For every configuration q = (r, z) of every trajectory in a data/sims/md/NOAu
# file, evaluate
#
#     λ_min(q) = min_{ω ∈ Ω} eigmin Λ(ω; q, T),     a(q) = min(0, λ_min(q)),
#
# with Ω = [Δω, ω_max] and ω_max = ħπ/Δt_MD (the MD Nyquist frequency, 8.27 eV
# at Δt = 0.25 fs). This is the input for the Markovian negative-tail estimate
# ΔE_a = ∫ v(t)ᵀ A(q(t)) v(t) dt of docs/negative_tail_markovian_bound.md, with
# A = a·𝟙 (bound) or A = a·e_min e_minᵀ (rank-1 estimate). The eigenvector
# e_min at ω* is stored for the second form.
#
# ω search (NegativeEigenvalue.jl, shared with verify_negative_eigenvalue_fig3.jl):
# coarse scan on Δω:Δω:ω_max, then Brent inside the bracket
# [ω_{i-1}, ω_{i+1}] around the coarse minimum. At 300 K (and 2000 K) λ_min(ω)
# has one broad negative dip (several eV wide) at every configuration checked;
# any other local minima are noise-level ripples (≲ 1e-4 of the dip depth). A
# 0.25 eV coarse step is therefore safe; it matches a 0.01 eV brute-force scan
# to ~1e-5 relative (verify_negative_eigenvalue_fig3.jl checks Fig. 3). Cost at
# 300 K ≈ 23 ms per configuration per core; the full sweep (39 files, 26 M
# configurations) is ≈ 170 core-hours.
#
# Parallelism: pmap over trajectories (Distributed workers), Threads.@threads
# over the configurations of one trajectory (threads of that worker).
#   local, threads only    julia --project -t 8 run_negative_eigenvalue_noau.jl --vib 16
#   local, workers×threads julia --project -p 4 -t 2 run_negative_eigenvalue_noau.jl --vib 16
#                          (-p needs --project, or the workers cannot load the package)
#   SLURM                  no -p; julia_build_procs() spawns SLURM_NTASKS workers
#                          with SLURM_CPUS_PER_TASK threads each
#   quick test             ... --vib 16 --translational_kinetic 1.0 --ntraj 4 --stride 20
#
# CLI: --vib <int> --translational_kinetic <float> --temperature <int>
#      --stride <int> --ntraj <int> --domega <float, eV> --omega_max <float, eV>
# With no arguments, sweeps every (vibrational_state, translational_kinetic)
# MD file with 1000 trajectories.
#
# Output: data/sims/friction/NOAu/negative/<savename>.h5, laid out like the MD
# files so load_md_trajectories(path; outputs = (:OutputPosition, :a, ...))
# reads it directly:
#   trajectory_<i>/Time            (N_i,)    MD time, au
#                 /OutputPosition  (2, N_i)  (r, z), bohr
#                 /a               (N_i,)    min(0, λ_min), au
#                 /lambda_min      (N_i,)    unclipped min_ω eigmin Λ, au
#                 /omega_star      (N_i,)    ω of the minimum, au (ħ = 1)
#                 /e_min           (2, N_i)  eigenvector at omega_star, e_z ≥ 0
# Root attributes record the MD source file, T, ω window, search settings,
# git commit and parallel layout. Results go to <savename>.h5.partial first,
# one chunk of trajectories at a time; a rerun resumes from the partial file,
# which is renamed to <savename>.h5 once every trajectory is done.

using Distributed
using DrWatson
@quickactivate "MemoryElectronicFriction"

using MemoryElectronicFriction
using Unitful, UnitfulAtomic
using HDF5
using Dates: now
using HokseonAssistant
HokseonAssistant.julia_build_procs()

# Per-configuration and per-trajectory work, defined on every worker.
include("NegativeEigenvalue.jl")

# ---------------------------------------------------------------------------
# CLI argument parsing. Falls back to the full sweep when run interactively.
# ---------------------------------------------------------------------------

let i = 1, _vib = nothing, _temp = nothing, _tk = nothing,
    _stride = nothing, _ntraj = nothing, _domega = nothing, _omega_max = nothing
    while i <= length(ARGS)
        if ARGS[i] == "--vib" && i < length(ARGS)
            _vib    = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--temperature" && i < length(ARGS)
            _temp   = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--translational_kinetic" && i < length(ARGS)
            _tk     = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--stride" && i < length(ARGS)
            _stride = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--ntraj" && i < length(ARGS)
            _ntraj  = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--domega" && i < length(ARGS)
            _domega = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--omega_max" && i < length(ARGS)
            _omega_max = parse(Float64, ARGS[i+1]); i += 2
        else
            i += 1
        end
    end
    global const CLI_VIB    = _vib
    global const CLI_TEMP   = _temp
    global const CLI_TK     = _tk
    global const CLI_STRIDE = _stride
    global const CLI_NTRAJ  = _ntraj
    global const CLI_DOMEGA = _domega
    global const CLI_OMEGA_MAX = _omega_max
end

# ---------------------------------------------------------------------------
# NOAu MD parameters. Must match run_md.jl exactly so dict_to_data_savename
# resolves to the same .h5 path the sweep wrote.
# ---------------------------------------------------------------------------

all_params_NOAu = Dict{String, Any}(
    "mass"                  => [(14.007 * 15.999 / (14.007 + 15.999)) * u"u"],   # μ_NO — POGO is 1-atom
    "r0"                    => [[1.15u"Å", 5.0u"Å"]],
    "translational_kinetic" => CLI_TK === nothing ?
                               [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0] .* u"eV" :
                               [CLI_TK * u"eV"],
    "state"                 => [1],
    "tmax"                  => [500.0u"fs"],
    "dt"                    => [0.25u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [2],
    "termination_threshold" => [5.0u"Å"],
    "vibrational_state"     => CLI_VIB === nothing ? [0, 3, 16] : [CLI_VIB],
    "trajectories"          => [1000],
)
params_list_NOAu = dict_list(all_params_NOAu)

# ---------------------------------------------------------------------------
# Negative-tail configuration. ω_max defaults to the MD Nyquist frequency
# ħπ/Δt; it is tied to the MD step, not to `stride`, which only subsamples
# configurations. For compressed bonds (r ≲ 1.03 Å) the minimum lies above
# 8.27 eV, so a there is the value at the window edge; --omega_max widens the
# window for a sensitivity check.
# ---------------------------------------------------------------------------

const NEG_config_noau = Dict{String, Any}(
    "model"        => :NOAu,
    "T_K"          => CLI_TEMP === nothing ? 300 : CLI_TEMP,
    "omega_max_eV" => CLI_OMEGA_MAX === nothing ?
                      nyquist_omega_max_eV(only(all_params_NOAu["dt"])) : CLI_OMEGA_MAX,
    "domega_eV"    => CLI_DOMEGA === nothing ? DEFAULT_DOMEGA_eV : CLI_DOMEGA,
    "omega_tol_eV" => DEFAULT_OMEGA_TOL_eV,
    "stride"       => CLI_STRIDE === nothing ? 1 : CLI_STRIDE,
)
CLI_NTRAJ === nothing || (NEG_config_noau["ntraj"] = CLI_NTRAJ)

# ---------------------------------------------------------------------------
# HDF5 output.
# ---------------------------------------------------------------------------

const DATASET_DESCRIPTIONS = [
    "Time"           => "MD time, atomic units (copied from the MD file)",
    "OutputPosition" => "configuration (r, z), bohr, size 2 × N (copied from the MD file)",
    "a"              => "min(0, lambda_min), atomic units of friction",
    "lambda_min"     => "min over ω ∈ [domega, omega_max] of the smallest eigenvalue of Λ(ω; q, T), not clipped (coarse-grid value where non-negative), atomic units",
    "omega_star"     => "ω at which lambda_min occurs, atomic units of energy (ħ = 1)",
    "e_min"          => "unit eigenvector of Λ(omega_star; q, T) belonging to lambda_min, (r, z) components, sign fixed so e_z ≥ 0",
]

function write_metadata!(fid, p, cfg, md_path, ntraj)
    A = attributes(fid)
    A["model"]               = String(cfg["model"])
    A["quantity"]            = "a(q) = min(0, min_ω eigmin Λ(ω; q, T)) along MD trajectories, ω ∈ [domega, omega_max]"
    A["md_file"]             = basename(md_path)
    A["n_trajectories"]      = ntraj
    A["md_dt_fs"]            = ustrip(u"fs", p["dt"])
    A["vibrational_state"]   = string(p["vibrational_state"])
    A["translational_kinetic_eV"] = ustrip(u"eV", p["translational_kinetic"])
    A["T_K"]                 = cfg["T_K"]
    A["omega_max_eV"]        = cfg["omega_max_eV"]
    A["domega_eV"]           = cfg["domega_eV"]
    A["omega_tol_eV"]        = cfg["omega_tol_eV"]
    A["stride"]              = cfg["stride"]
    A["omega_search"]        = "coarse scan domega:domega:omega_max, then Brent (abs_tol = omega_tol) in the bracket around the coarse minimum"
    A["units"]               = "atomic units throughout"
    A["git_commit"]          = something(gitdescribe(projectdir()), "unknown")
    A["created"]             = string(now())
    A["julia_version"]       = string(VERSION)
    A["nworkers"]            = nworkers()
    A["nthreads_per_worker"] = remotecall_fetch(Threads.nthreads, first(workers()))
    for (name, desc) in DATASET_DESCRIPTIONS
        A["dataset_$name"] = desc
    end
end

function write_trajectory!(fid, i, res)
    name = "trajectory_$i"
    haskey(fid, name) && delete_object(fid, name)   # half-written group from a crash
    g = create_group(fid, name)
    for (dset, _) in DATASET_DESCRIPTIONS
        g[dset] = getproperty(res, Symbol(dset))
    end
end

# A group is complete once its last dataset (e_min) exists.
completed_trajectories(path) = h5open(path, "r") do f
    Set(k for k in keys(f) if haskey(f[k], "e_min"))
end

# ---------------------------------------------------------------------------
# Driver: one output file per MD parameter set.
# ---------------------------------------------------------------------------

function run_negative_eigenvalue(cfg, params_list)

    @unpack model, T_K, omega_max_eV, domega_eV, omega_tol_eV, stride = cfg

    T_au     = austrip(T_K * u"K")
    ω_coarse = coarse_omega_grid(domega_eV, omega_max_eV)
    ω_tol    = austrip(omega_tol_eV * u"eV")
    ads      = NOAuAdsorbate()

    for p in params_list
        md_path = datadir(dict_to_data_savename(p, String(model))...)
        if !isfile(md_path)
            @warn "MD file not found, skipping" md_path
            continue
        end

        savingpath, savingname = negative_tail_dict_to_data_savename(p, cfg)
        full_data_path = datadir(savingpath, savingname)
        partial_path   = full_data_path * ".partial"
        if isfile(full_data_path)
            @info "Skipping (already saved)" full_data_path
            continue
        end

        trajs = load_md_trajectories(md_path; outputs = (:OutputPosition,))
        ntraj = min(get(cfg, "ntraj", length(trajs)), length(trajs))

        if isfile(partial_path)
            done = completed_trajectories(partial_path)
        else
            h5open(f -> write_metadata!(f, p, cfg, md_path, ntraj), partial_path, "w")
            done = Set{String}()
        end
        todo = [i for i in 1:ntraj if "trajectory_$i" ∉ done]
        n_conf = sum(length(1:stride:length(trajs[i].t)) for i in todo; init = 0)
        @info "Negative-tail eigenvalue" md_file=basename(md_path) T_K vib=p["vibrational_state"] Ek=p["translational_kinetic"] n_todo=length(todo) n_done=length(done) n_conf n_omega_coarse=length(ω_coarse) nworkers=nworkers()

        # Chunks of trajectories: each chunk is pmap'ed, then written to the
        # partial file, so a killed job loses at most one chunk.
        t0 = time()
        n_finished = 0
        for batch in Iterators.partition(todo, 4 * nworkers())
            # Only (t, Q) travel to the workers, not the whole trajectory list.
            inputs  = [(trajs[i].t[1:stride:end], trajs[i].OutputPosition[:, 1:stride:end]) for i in batch]
            results = pmap(x -> negative_tail_along_trajectory(x..., ads, T_au, ω_coarse, ω_tol), inputs)
            h5open(partial_path, "r+") do f
                foreach(((i, r),) -> write_trajectory!(f, i, r), zip(batch, results))
            end
            n_finished += length(batch)
            elapsed = time() - t0
            @info "Progress" done="$(n_finished)/$(length(todo))" elapsed_min=round(elapsed / 60; digits = 1) eta_min=round(elapsed / n_finished * (length(todo) - n_finished) / 60; digits = 1)
        end

        mv(partial_path, full_data_path)
        summarize(full_data_path, omega_max_eV, domega_eV)
    end
end

# One-line health check per file: how often a < 0, how negative, where ω* sits,
# and how many minima landed on the upper edge of the ω window.
function summarize(path, omega_max_eV, domega_eV)
    a, ω = h5open(path, "r") do f
        ks = [k for k in keys(f)]
        reduce(vcat, [read(f[k]["a"]) for k in ks]), reduce(vcat, [read(f[k]["omega_star"]) for k in ks])
    end
    ω_eV = ustrip.(auconvert.(u"eV", ω))
    @info "Saved negative-tail eigenvalues" path n_conf=length(a) frac_negative=count(<(0), a) / length(a) a_min_au=minimum(a) omega_star_eV=extrema(ω_eV) n_at_omega_max=count(>(omega_max_eV - domega_eV), ω_eV)
end

run_negative_eigenvalue(NEG_config_noau, params_list_NOAu)
