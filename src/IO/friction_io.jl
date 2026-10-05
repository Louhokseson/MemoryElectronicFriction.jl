# ------------------------------------------------------------------------------
# Negative-tail friction filename construction.
#
# a(q) = min(0, min_ω λ_min Λ(ω; q, T)) evaluated along MD trajectories
# (scripts/compute/friction/NOAu/run_negative_eigenvalue_noau.jl). Same MD key
# subset as the CPA files, plus the settings that change the numbers. The
# optional `ntraj` key only appears for partial (test) runs.
# ------------------------------------------------------------------------------

const _NEG_CFG_KEYS = ("T_K", "omega_max_eV", "domega_eV", "stride", "ntraj")

function negative_tail_dict_to_data_savename(p::Dict{String,Any}, cfg::Dict{String,Any})
    md_part  = Dict{String,Any}(k => p[k] for k in _CPA_MD_KEYS if haskey(p, k))
    cfg_part = Dict{String,Any}(k => cfg[k] for k in _NEG_CFG_KEYS if haskey(cfg, k))
    merged   = merge(_sanitize_for_savename(md_part), _sanitize_for_savename(cfg_part))
    savingpath = joinpath("sims", "friction", String(cfg["model"]), "negative")
    isdir(datadir(savingpath)) || mkpath(datadir(savingpath))
    savingname = savename(merged, "h5")
    return (savingpath, savingname)
end
