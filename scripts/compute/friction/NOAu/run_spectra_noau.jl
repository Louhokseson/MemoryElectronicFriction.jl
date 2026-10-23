# run_spectra_noau.jl — friction spectra Λ(ω; q, T) along the first trajectories
# of the NO/Au classical MD runs.
#
# For every configuration q = (r, z) of the first `ntraj` (default 10)
# trajectories of a data/sims/md/NOAu file, evaluate the 2×2 friction tensor
# Λ(ω; q, T) on the memory-CPA ω grid and store its three independent elements
# Λ_rr, Λ_zz, Λ_rz (Λ is symmetric, Λ_zr = Λ_rz).
#
# ω grid: DEFAULT_ω_GRID_eV of ../../cpa/DeltaE.jl (0, ten log-spaced points
# 1e-7 … 3.16e-3 eV, then 0.01:0.01:20 eV; 2011 points), so the spectra can be
# cosine-transformed exactly as the memory CPA does. Keep the two in sync.
#
# Engine: friction_tensor from NegativeEigenvalue.jl, which splits the energy
# integral at the Fermi steps and resonances. FrequencyLambda.Lambda integrates
# (−∞, ∞) in one piece and at 300 K occasionally misses part of the Fermi
# window (see NegativeEigenvalue.jl); such a miss would be a spike in Λ(ω).
#
# Parallelism: pmap over the configurations of one trajectory (Distributed
# workers), Threads.@threads over ω at each configuration (threads of that
# worker).
#   local, threads only    julia --project -t 8 run_spectra_noau.jl --vib 16
#   local, workers×threads julia --project -p 4 -t 2 run_spectra_noau.jl --vib 16
#                          (-p needs --project, or the workers cannot load the package)
#   SLURM                  no -p; julia_build_procs() spawns SLURM_NTASKS workers
#                          with SLURM_CPUS_PER_TASK threads each
#   quick test             ... --vib 16 --translational_kinetic 1.0 --ntraj 1 --stride 100
#
# CLI: --vib <int>[,<int>...] --translational_kinetic <float> --temperature <int>
#      --ntraj <int> --stride <int>
# With no arguments, sweeps every (vibrational_state, translational_kinetic)
# MD file with 1000 trajectories, at 300 K, 10 trajectories, every MD step.
#
# Output: data/sims/friction/NOAu/spectra/<savename>.h5, laid out like the MD
# files so load_md_trajectories(path; outputs = (:OutputPosition, :omega,
# :Lambda_rr, :Lambda_zz, :Lambda_rz)) reads it directly:
#   trajectory_<i>/Time            (N_i,)      MD time, au
#                 /OutputPosition  (2, N_i)    (r, z), bohr
#                 /omega           (Nω,)       ω grid, au (ħ = 1)
#                 /Lambda_rr       (Nω, N_i)   Λ_rr(ω; q(t)), au
#                 /Lambda_zz       (Nω, N_i)   Λ_zz(ω; q(t)), au
#                 /Lambda_rz       (Nω, N_i)   Λ_rz(ω; q(t)), au
# Column n of each Lambda_* is the spectrum at Time[n]. The three components
# are separate 2D datasets because load_md_trajectories flattens any 3D
# dataset. Root attributes record the MD source file, T, ω grid, git commit
# and parallel layout. Results go to <savename>.h5.partial first, one
# trajectory at a time; a rerun resumes from the partial file, which is
# renamed to <savename>.h5 once every trajectory is done.

using Distributed
using DrWatson
@quickactivate "MemoryElectronicFriction"

using MemoryElectronicFriction
using Unitful, UnitfulAtomic
using HDF5
using Dates: now
using HokseonAssistant
HokseonAssistant.julia_build_procs()

# friction_tensor(ω, ads, q, T), defined on every worker.
include("NegativeEigenvalue.jl")

@everywhere begin
    # Spectrum at one configuration: Nω × 3 matrix, columns (rr, zz, rz).
    function spectrum_at_configuration(q, ads, T, ω_grid)
        Λ = zeros(length(ω_grid), 3)
        Threads.@threads for j in eachindex(ω_grid)
            L = friction_tensor(ω_grid[j], ads, q, T)
            Λ[j, 1], Λ[j, 2], Λ[j, 3] = L[1, 1], L[2, 2], L[1, 2]
        end
        return Λ
    end
end

# ---------------------------------------------------------------------------
# CLI argument parsing. Falls back to the full sweep when run interactively.
# ---------------------------------------------------------------------------

let i = 1, _vib = nothing, _temp = nothing, _tk = nothing, _ntraj = nothing, _stride = nothing
    while i <= length(ARGS)
        if ARGS[i] == "--vib" && i < length(ARGS)
            _vib    = parse.(Int, split(ARGS[i+1], ',')); i += 2   # e.g. 16 or 0,3
        elseif ARGS[i] == "--temperature" && i < length(ARGS)
            _temp   = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--translational_kinetic" && i < length(ARGS)
            _tk     = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--ntraj" && i < length(ARGS)
            _ntraj  = parse(Int, ARGS[i+1]); i += 2
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
# resolves to the same .h5 path the sweep wrote. MD files that do not exist
# on this machine are skipped with a warning.
# ---------------------------------------------------------------------------

all_params_NOAu = Dict{String, Any}(
    "mass"                  => [[14.007 * 15.999 / (14.007 + 15.999), 14.007 + 15.999] .* u"u"],   # [μ_NO, M_NO]: r carries μ, z carries M
    "r0"                    => [[1.15u"Å", 5.0u"Å"]],
    "translational_kinetic" => CLI_TK === nothing ?
                               [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0] .* u"eV" :
                               [CLI_TK * u"eV"],
    "state"                 => [1],
    "tmax"                  => [1000.0u"fs"],
    "dt"                    => [0.25u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [2],
    "termination_threshold" => [5.0u"Å"],
    "vibrational_state"     => CLI_VIB === nothing ? [0, 3, 16] : CLI_VIB,
    "trajectories"          => [1000],
)
params_list_NOAu = dict_list(all_params_NOAu)

# ---------------------------------------------------------------------------
# Spectra configuration. "folder" sends the output to .../friction/NOAu/spectra;
# T_K, stride and ntraj go into the file name.
# ---------------------------------------------------------------------------

const OMEGA_GRID_eV =
    vcat(0.0,
         [1e-7, 3.16e-7, 1e-6, 3.16e-6, 1e-5, 3.16e-5,
          1e-4, 3.16e-4, 1e-3, 3.16e-3],
         collect(0.01:0.01:20.0))

const SPEC_config_noau = Dict{String, Any}(
    "model"  => :NOAu,
    "folder" => "spectra",
    "T_K"    => CLI_TEMP   === nothing ? 300 : CLI_TEMP,
    "ntraj"  => CLI_NTRAJ  === nothing ? 10  : CLI_NTRAJ,
    "stride" => CLI_STRIDE === nothing ? 1   : CLI_STRIDE,
)

# ---------------------------------------------------------------------------
# HDF5 output.
# ---------------------------------------------------------------------------

const DATASET_DESCRIPTIONS = [
    "Time"           => "MD time, atomic units (copied from the MD file)",
    "OutputPosition" => "configuration (r, z), bohr, size 2 × N (copied from the MD file)",
    "omega"          => "ω grid, atomic units of energy (ħ = 1), size Nω",
    "Lambda_rr"      => "Λ_rr(ω; q, T), atomic units of friction, size Nω × N (column n ↔ Time[n])",
    "Lambda_zz"      => "Λ_zz(ω; q, T), atomic units of friction, size Nω × N (column n ↔ Time[n])",
    "Lambda_rz"      => "Λ_rz(ω; q, T) = Λ_zr, atomic units of friction, size Nω × N (column n ↔ Time[n])",
]

function write_metadata!(fid, p, cfg, md_path, ntraj)
    A = attributes(fid)
    A["model"]               = String(cfg["model"])
    A["quantity"]            = "friction spectra Λ(ω; q, T) (rr, zz, rz) along MD trajectories"
    A["md_file"]             = basename(md_path)
    A["n_trajectories"]      = ntraj
    A["md_dt_fs"]            = ustrip(u"fs", p["dt"])
    A["vibrational_state"]   = string(p["vibrational_state"])
    A["translational_kinetic_eV"] = ustrip(u"eV", p["translational_kinetic"])
    A["T_K"]                 = cfg["T_K"]
    A["stride"]              = cfg["stride"]
    A["omega_grid"]          = "DEFAULT_ω_GRID_eV of scripts/compute/cpa/DeltaE.jl: 0, ten log-spaced points 1e-7 … 3.16e-3 eV, then 0.01:0.01:20 eV"
    A["n_omega"]             = length(OMEGA_GRID_eV)
    A["engine"]              = "friction_tensor of scripts/compute/friction/NOAu/NegativeEigenvalue.jl (energy integral split at the Fermi steps and resonances, rtol = 1e-6)"
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

# A group is complete once its last dataset (Lambda_rz) exists.
completed_trajectories(path) = h5open(path, "r") do f
    Set(k for k in keys(f) if haskey(f[k], "Lambda_rz"))
end

# ---------------------------------------------------------------------------
# One trajectory: t (N,), Q (2, N) in au. pmap over configurations.
# ---------------------------------------------------------------------------

function spectra_along_trajectory(t, Q, ads, T, ω_grid)
    qs = [SA[Q[1, n], Q[2, n]] for n in axes(Q, 2)]
    Λs = pmap(q -> spectrum_at_configuration(q, ads, T, ω_grid), qs)
    Lambda_rr, Lambda_zz, Lambda_rz = (zeros(length(ω_grid), length(qs)) for _ in 1:3)
    for (n, Λ) in enumerate(Λs)
        Lambda_rr[:, n] .= @view Λ[:, 1]
        Lambda_zz[:, n] .= @view Λ[:, 2]
        Lambda_rz[:, n] .= @view Λ[:, 3]
    end
    return (; Time = t, OutputPosition = Q, omega = ω_grid, Lambda_rr, Lambda_zz, Lambda_rz)
end

# ---------------------------------------------------------------------------
# Driver: one output file per MD parameter set.
# ---------------------------------------------------------------------------

function run_spectra(cfg, params_list)

    @unpack model, T_K, ntraj, stride = cfg

    T_au = austrip(T_K * u"K")
    ω_au = austrip.(OMEGA_GRID_eV .* u"eV")
    ads  = NOAuAdsorbate()

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
        n     = min(ntraj, length(trajs))

        if isfile(partial_path)
            done = completed_trajectories(partial_path)
        else
            h5open(f -> write_metadata!(f, p, cfg, md_path, n), partial_path, "w")
            done = Set{String}()
        end
        todo = [i for i in 1:n if "trajectory_$i" ∉ done]
        n_conf = sum(length(1:stride:length(trajs[i].t)) for i in todo; init = 0)
        @info "Friction spectra" md_file=basename(md_path) T_K vib=p["vibrational_state"] Ek=p["translational_kinetic"] n_todo=length(todo) n_done=length(done) n_conf n_omega=length(ω_au) nworkers=nworkers()

        t0 = time()
        for (k, i) in enumerate(todo)
            res = spectra_along_trajectory(trajs[i].t[1:stride:end],
                                           trajs[i].OutputPosition[:, 1:stride:end],
                                           ads, T_au, ω_au)
            h5open(f -> write_trajectory!(f, i, res), partial_path, "r+")
            elapsed = time() - t0
            @info "Progress" trajectory=i done="$(k)/$(length(todo))" n_conf=length(res.Time) elapsed_min=round(elapsed / 60; digits = 1) eta_min=round(elapsed / k * (length(todo) - k) / 60; digits = 1)
        end

        mv(partial_path, full_data_path)
        summarize(full_data_path)
    end
end

# One-line health check per file: configuration count, non-finite entries,
# and the range of the Markovian (ω = 0) diagonal elements.
function summarize(path)
    rr0, zz0, n_bad = h5open(path, "r") do f
        gs = [f[k] for k in keys(f)]
        rr0 = reduce(vcat, [read(g["Lambda_rr"])[1, :] for g in gs])
        zz0 = reduce(vcat, [read(g["Lambda_zz"])[1, :] for g in gs])
        n_bad = sum(count(!isfinite, read(g[d])) for g in gs for d in ("Lambda_rr", "Lambda_zz", "Lambda_rz"))
        rr0, zz0, n_bad
    end
    @info "Saved friction spectra" path n_conf=length(rr0) n_nonfinite=n_bad Lambda_rr_omega0_au=extrema(rr0) Lambda_zz_omega0_au=extrema(zz0)
end

run_spectra(SPEC_config_noau, params_list_NOAu)
