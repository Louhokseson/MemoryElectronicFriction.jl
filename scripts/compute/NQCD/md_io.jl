using DrWatson
using HDF5
using Unitful

# -----------------------------------------------------------------------------
# Filename construction. Mirrors HonGeAnalysis' dict_to_data_savename but
# simpler: no distributed job IDs, no method key. `savename` gets a sanitized
# copy of the param dict so Unitful quantities and 1-element vectors collapse
# to plain numbers in the filename.
# -----------------------------------------------------------------------------

_savename_value(v::Unitful.Quantity) = ustrip(v)
_savename_value(v::AbstractVector{<:Unitful.Quantity}) =
    length(v) == 1 ? ustrip(v[1]) : join(string.(ustrip.(v)), "-")
_savename_value(v::AbstractVector) = length(v) == 1 ? v[1] : v
_savename_value(::Nothing) = "off"
_savename_value(v) = v

function _sanitize_for_savename(param_dict::Dict{String,Any})
    Dict{String,Any}(k => _savename_value(v) for (k, v) in param_dict)
end

function dict_to_data_savename(param_dict::Dict{String,Any}, model_folder::AbstractString)
    savingpath = joinpath("sims", "md", model_folder)
    isdir(datadir(savingpath)) || mkpath(datadir(savingpath))
    savingname = savename(_sanitize_for_savename(param_dict), "h5")
    return (savingpath, savingname)
end

# -----------------------------------------------------------------------------
# Trajectory loading. NQCDynamics' FileReduction lays each .h5 file out as
#   trajectory_<i>/Time            (Nt_i,)
#                 /OutputPosition  (D, 1, Nt_i)
#                 /OutputVelocity  (D, 1, Nt_i)
#                 /Output{Kinetic,Potential,Total}Energy (Nt_i,)
# D = 1 for ErpenbeckThoss (1-DOF), D = 2 for NOAu (bond, z). Trajectory
# lengths Nt_i differ when the TerminatingCallback fires at different times,
# so we return a Vector (one element per trajectory), not a rectangular array.
#
# Each element is a NamedTuple keyed by `:t` plus whatever outputs the caller
# requested via the `outputs` keyword — defaults to position+velocity, but you
# can pass e.g. `outputs = (:OutputPosition, :OutputKineticEnergy)` to read
# only those, or add energies. The singleton "atoms" axis is squeezed for 3D
# arrays (so position/velocity come back as D × Nt) and 1D arrays (energies)
# pass through unchanged.
# -----------------------------------------------------------------------------

const _DEFAULT_OUTPUTS = (:OutputPosition, :OutputVelocity)

_traj_index(name::AbstractString) = parse(Int, last(split(name, '_')))

function _read_field(g, name::AbstractString)
    a = read(g[name])
    return ndims(a) == 3 ? dropdims(a; dims = 2) : a
end

function _read_trajectory(g, outputs)
    t = read(g["Time"])
    return (; t, (o => _read_field(g, String(o)) for o in outputs)...)
end

function load_md_trajectories(filepath::AbstractString;
                              outputs = _DEFAULT_OUTPUTS)
    isfile(filepath) || error("MD file not found: $filepath")
    h5open(filepath, "r") do f
        names = sort(collect(keys(f)); by = _traj_index)
        return [_read_trajectory(f[n], outputs) for n in names]
    end
end

function load_md_trajectories(params::Dict{String,Any}, model_folder::AbstractString;
                              outputs = _DEFAULT_OUTPUTS)
    savingpath, savingname = dict_to_data_savename(params, model_folder)
    return load_md_trajectories(datadir(savingpath, savingname); outputs)
end

# Convenience: number of degrees of freedom in a trajectory. Requires that
# `:OutputPosition` is among the loaded outputs.
ndofs(traj::NamedTuple) = size(traj.OutputPosition, 1)
