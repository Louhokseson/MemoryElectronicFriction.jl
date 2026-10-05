# run_negative_bound_CPA_noau.jl — rigorous Markovian bound on the energy the
# negative tail of Λ(ω) can return to NO, along the NO/Au MD trajectories.
#
# docs/negative_tail_markovian_bound.md, Sec. 3 (isotropic form):
#
#     ΔE_a = ∫₀ᵀ a(q(t)) |v(t)|² dt  ≈  Δt Σᵢ a(qᵢ) |vᵢ|²  ≤  ΔE₋  ≤  0,
#
# i.e. the Markovian CPA with η(q) replaced by a(q)·𝟙, on the same time
# discretisation as the Markovian delta_energy in ../DeltaE.jl. −ΔE_a is the
# largest energy the negative tail of Λ(ω) can feed back to the molecule
# (frozen-q / adiabatic bound; caveats in the note).
#
# Inputs: a(q) from run_negative_eigenvalue_noau.jl
# (data/sims/friction/NOAu/negative) and v from the matching MD file. The CLI
# mirrors that script, so the same options find the same a(q) file:
#   --vib <int>[,<int>...] --translational_kinetic <float> --temperature <int>
#   --stride <int> --ntraj <int> --domega <float, eV> --omega_max <float, eV>
# MD files whose a(q) file does not exist yet are skipped.
#
# Output: data/sims/cpa/NOAu/negative_bound/<savename>.h5, same layout as the
# memory / Markovian CPA files:
#   DeltaE_au  (2, n_traj)  rows [ΔE_r; ΔE_z], ΔE_k = Δt Σᵢ a(qᵢ) v_k,i².
#              sum over rows = ΔE_a; the z row alone is ∫ a ż² dt.
# A summary table (meV) at the end sets ⟨ΔE_a⟩ next to the 300 K memory and
# Markovian CPA ⟨ΔE⟩ where those files exist.
#
# A sum per trajectory: seconds per file on a laptop, no workers needed.
#   julia --project scripts/compute/cpa/NOAu/run_negative_bound_CPA_noau.jl

using Distributed
using DrWatson
@quickactivate "MemoryElectronicFriction"

using MemoryElectronicFriction
using Unitful, UnitfulAtomic
using HDF5
using Printf
using Statistics: mean

# Search defaults (DEFAULT_DOMEGA_eV, nyquist_omega_max_eV) of the a(q) run.
include("../../friction/NOAu/NegativeEigenvalue.jl")

# ---------------------------------------------------------------------------
# CLI argument parsing (same options as run_negative_eigenvalue_noau.jl).
# ---------------------------------------------------------------------------

let i = 1, _vib = nothing, _temp = nothing, _tk = nothing,
    _stride = nothing, _ntraj = nothing, _domega = nothing, _omega_max = nothing
    while i <= length(ARGS)
        if ARGS[i] == "--vib" && i < length(ARGS)
            _vib    = parse.(Int, split(ARGS[i+1], ',')); i += 2   # e.g. 16 or 0,3
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
    "vibrational_state"     => CLI_VIB === nothing ? [0, 3, 16] : CLI_VIB,
    "trajectories"          => [1000],
)
params_list_NOAu = dict_list(all_params_NOAu)

# ---------------------------------------------------------------------------
# Configurations. NEG_config must equal the one in run_negative_eigenvalue_noau.jl
# (it locates the a(q) file); CPA_config names the output.
# ---------------------------------------------------------------------------

const NEG_config_noau = Dict{String, Any}(
    "model"        => :NOAu,
    "T_K"          => CLI_TEMP === nothing ? 300 : CLI_TEMP,
    "omega_max_eV" => CLI_OMEGA_MAX === nothing ?
                      nyquist_omega_max_eV(only(all_params_NOAu["dt"])) : CLI_OMEGA_MAX,
    "domega_eV"    => CLI_DOMEGA === nothing ? DEFAULT_DOMEGA_eV : CLI_DOMEGA,
    "stride"       => CLI_STRIDE === nothing ? 1 : CLI_STRIDE,
)
CLI_NTRAJ === nothing || (NEG_config_noau["ntraj"] = CLI_NTRAJ)

const CPA_config_noau = merge(NEG_config_noau, Dict{String, Any}("variant" => "negative_bound"))

# 300 K memory / Markovian CPA results, for the summary table only.
const REFERENCE_configs = [
    "memory"    => Dict{String, Any}("model" => :NOAu, "T_K" => NEG_config_noau["T_K"],
                                     "stride" => 1, "kernel_average" => :arithmetic),
    "markovian" => Dict{String, Any}("model" => :NOAu, "T_K" => NEG_config_noau["T_K"],
                                     "stride" => 1),
]

# ---------------------------------------------------------------------------
# ΔE_a for one trajectory: Δt Σᵢ a(qᵢ) v_k,i², per DOF k.
# ---------------------------------------------------------------------------

function delta_energy_negative_bound(neg_traj, md_traj, stride::Integer, dt_au::Real)
    idx = 1:stride:length(md_traj.t)
    V   = md_traj.OutputVelocity[:, idx]
    # The a(q) file must sample exactly these MD configurations.
    md_traj.t[idx] == neg_traj.t &&
        md_traj.OutputPosition[:, idx] == neg_traj.OutputPosition ||
        error("a(q) file and MD trajectory do not sample the same configurations")
    Δt = stride * dt_au
    return [Δt * sum(neg_traj.a .* abs2.(V[k, :])) for k in axes(V, 1)]
end

# ⟨Σ_k ΔE_k⟩ over the first n trajectories of a CPA file, in meV, or nothing.
function mean_cpa_meV(path, n)
    isfile(path) || return nothing
    ΔE = h5open(f -> read(f["DeltaE_au"]), path)
    size(ΔE, 2) >= n || return nothing
    return 1e3 * mean(sum(ΔE[:, 1:n]; dims = 1)) * ustrip(auconvert(u"eV", 1))
end

# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

function run_negative_bound_CPA(neg_cfg, cpa_cfg, params_list)
    stride = neg_cfg["stride"]
    to_meV = 1e3 * ustrip(auconvert(u"eV", 1))
    rows   = []

    for p in params_list
        md_path  = datadir(dict_to_data_savename(p, String(neg_cfg["model"]))...)
        neg_path = datadir(negative_tail_dict_to_data_savename(p, neg_cfg)...)
        if !isfile(neg_path)
            @warn "a(q) file not found (not computed yet?), skipping" neg_path
            continue
        end

        savingpath, savingname = CPA_dict_to_data_savename(p, cpa_cfg)
        full_data_path = datadir(savingpath, savingname)

        # Reuse a saved result only if it is newer than its a(q) file, so a
        # recomputed a(q) never leaves a stale ΔE_a behind.
        ΔE_au_mat = if isfile(full_data_path) && mtime(full_data_path) > mtime(neg_path)
            @info "Already saved, reading for the summary" full_data_path
            h5open(f -> read(f["DeltaE_au"]), full_data_path)
        else
            neg_trajs = load_md_trajectories(neg_path; outputs = (:OutputPosition, :a))
            md_trajs  = load_md_trajectories(md_path)
            dt_au     = austrip(p["dt"])
            ΔE = reduce(hcat, [delta_energy_negative_bound(neg_trajs[i], md_trajs[i], stride, dt_au)
                               for i in eachindex(neg_trajs)])   # D × n_traj

            h5open(full_data_path, "w") do fid
                fid["DeltaE_au"] = ΔE
                A = attributes(fid)
                A["quantity"]      = "ΔE_a = Δt Σᵢ a(qᵢ) |vᵢ|² (isotropic negative-tail bound, docs/negative_tail_markovian_bound.md); rows [r; z], au (Hartree)"
                A["negative_file"] = basename(neg_path)
                A["md_file"]       = basename(md_path)
                A["T_K"]           = neg_cfg["T_K"]
                A["omega_max_eV"]  = neg_cfg["omega_max_eV"]
                A["domega_eV"]     = neg_cfg["domega_eV"]
                A["stride"]        = stride
                A["git_commit"]    = something(gitdescribe(projectdir()), "unknown")
            end
            @info "Saved negative-tail bound" full_data_path size=size(ΔE)
            ΔE
        end

        n   = size(ΔE_au_mat, 2)
        tot = vec(sum(ΔE_au_mat; dims = 1)) .* to_meV
        ref = Dict(name => mean_cpa_meV(datadir(CPA_dict_to_data_savename(p, cfg)...), n)
                   for (name, cfg) in REFERENCE_configs)
        push!(rows, (; vib = p["vibrational_state"], Ek = ustrip(u"eV", p["translational_kinetic"]), n,
                       mean_a = mean(tot), min_a = minimum(tot),
                       z_share = sum(ΔE_au_mat[2, :]) / sum(ΔE_au_mat),
                       mem = ref["memory"], mark = ref["markovian"]))
    end

    print_summary(rows, neg_cfg["T_K"])
end

fmt(x) = x === nothing ? "      –" : @sprintf("%7.2f", x)

function print_summary(rows, T_K)
    isempty(rows) && return
    println("\nNegative-tail bound ΔE_a = Δt Σ a|v|² at $(T_K) K (meV; ΔE < 0 = energy returned to NO)")
    println(" vib    Ek   n_traj   ⟨ΔE_a⟩   min ΔE_a   z share │ ⟨ΔE_mem⟩   ⟨ΔE_M⟩   |⟨ΔE_a⟩|/⟨ΔE_mem⟩")
    for r in rows
        ratio = r.mem === nothing ? "      –" : @sprintf("%7.4f", abs(r.mean_a) / r.mem)
        @printf(" %3d  %4.1f  %7d  %s    %s    %5.3f  │  %s  %s   %s\n",
                r.vib, r.Ek, r.n, fmt(r.mean_a), fmt(r.min_a), r.z_share, fmt(r.mem), fmt(r.mark), ratio)
    end
end

run_negative_bound_CPA(NEG_config_noau, CPA_config_noau, params_list_NOAu)
