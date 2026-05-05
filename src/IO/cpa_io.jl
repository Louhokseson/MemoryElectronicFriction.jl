# ------------------------------------------------------------------------------
# CPA (Classical Path Approximation) filename construction.
#
# CPA results are identified by a subset of the MD parameters plus a subset of
# the CPA configuration keys.  We deliberately drop keys that are either
# invariant across a sweep (termination_*) or runtime details (parallel, ω) so
# that the savename stays well under OS filename-length limits.
# ------------------------------------------------------------------------------

const _CPA_MD_KEYS = ("mass", "Γ", "r0", "translational_kinetic", "state",
                      "dt", "vibrational_state", "trajectories")
const _CPA_CFG_KEYS = ("model", "T_K", "stride", "kernel_average")

function CPA_dict_to_data_savename(p::Dict{String,Any}, cfg::Dict{String,Any})
    md_part  = Dict{String,Any}(k => get(p, k, nothing) for k in _CPA_MD_KEYS if haskey(p, k))
    cfg_part = Dict{String,Any}(k => cfg[k] for k in _CPA_CFG_KEYS if haskey(cfg, k))
    merged   = merge(_sanitize_for_savename(md_part), _sanitize_for_savename(cfg_part))
    savingpath = joinpath("sims", "cpa", String(cfg["model"]))
    isdir(datadir(savingpath)) || mkpath(datadir(savingpath))
    savingname = savename(merged, "h5")
    return (savingpath, savingname)
end
