using DrWatson
@quickactivate "HokseonReproduce"

if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end

using NQCModels
using NQCDynamics
using LinearAlgebra: eigen
using Unitful, UnitfulAtomic
using HokseonAssistant
HokseonAssistant.julia_build_procs()

# -----------------------------------------------------------------------------
# Born-Oppenheimer classical MD on a NQCModels QuantumModel.
# See nqcd_md.jl for the rationale behind the local BOAdiabaticModel wrapper.
# -----------------------------------------------------------------------------

struct BOAdiabaticModel{M<:NQCModels.QuantumModels.QuantumModel} <: NQCModels.ClassicalModels.ClassicalModel
    quantum_model::M
    state::Int
end

NQCModels.ndofs(m::BOAdiabaticModel) = NQCModels.ndofs(m.quantum_model)

function NQCModels.potential(m::BOAdiabaticModel, r::AbstractMatrix)
    V = NQCModels.potential(m.quantum_model, r)
    eig = eigen(V)
    return eig.values[sortperm(eig.values)][m.state]
end

function NQCModels.derivative!(m::BOAdiabaticModel, output::AbstractMatrix, r::AbstractMatrix)
    V = NQCModels.potential(m.quantum_model, r)
    eig  = eigen(V)
    perm = sortperm(eig.values)
    U = eig.vectors[:, perm]

    D = NQCModels.zero_derivative(m.quantum_model, r)
    NQCModels.derivative!(m.quantum_model, D, r)

    for I in eachindex(output, D)
        output[I] = (U' * D[I] * U)[m.state, m.state]
    end
    return output
end

# `state` is deliberately untyped so the @unpack of an Any-valued dict entry
# doesn't trip the linter; BOAdiabaticModel converts it at construction.
function run_BO_dynamics(quantum_model, atoms, r0, v0, full_data_path;
                         state     = 2,
                         tspan     = (0.0, austrip(500u"fs")),
                         dt        = austrip(0.01u"fs"),
                         terminate = (u, t, _) -> false)

    bo_model = BOAdiabaticModel(quantum_model, state)
    sim      = Simulation(atoms, bo_model)
    dist     = DynamicalDistribution(v0, r0, size(r0))

    return run_dynamics(sim, tspan, dist; dt,
        callback = DynamicsUtils.TerminatingCallback(terminate),
        output = (OutputPosition, OutputVelocity,
                  OutputKineticEnergy, OutputPotentialEnergy, OutputTotalEnergy),
        trajectories = 1,
        reduction    = FileReduction(full_data_path))
end

function run_erpenbeck_thoss(params, full_data_path)
    @unpack mass, Γ, r0, translational_kinetic, state, tmax, dt,
            termination_min_time, termination_coord_idx, termination_threshold = params

    atoms  = Atoms(mass)
    m      = atoms.masses[1]
    r0_mat = reshape(austrip.(r0), length(r0), 1)
    # 1 DOF: send the bond coordinate inward (-r) with all the incident KE.
    v0     = fill(-sqrt(2 * austrip(translational_kinetic) / m), 1, 1)
    model  = NQCModels.ErpenbeckThoss(; Γ = austrip(Γ))

    t_min  = austrip(termination_min_time)
    thresh = austrip(termination_threshold)
    terminate = (u, t, _) -> t > t_min &&
                             DynamicsUtils.get_positions(u)[termination_coord_idx, 1] > thresh

    return run_BO_dynamics(model, atoms, r0_mat, v0, full_data_path;
                           state, tspan = (0.0, austrip(tmax)),
                           dt = austrip(dt), terminate)
end

function run_pogo(params, full_data_path)
    @unpack mass, Γ, r0, translational_kinetic, state, tmax, dt,
            termination_min_time, termination_coord_idx, termination_threshold = params

    atoms  = Atoms(mass)
    m      = atoms.masses[1]
    r0_mat = reshape(austrip.(r0), length(r0), 1)
    # 2 DOFs (r, z): put all KE into incoming z-motion (toward surface, -z).
    v0        = zeros(2, 1)
    v0[2, 1]  = -sqrt(2 * austrip(translational_kinetic) / m)
    model     = POGOModel(; Γ = austrip(Γ))

    t_min  = austrip(termination_min_time)
    thresh = austrip(termination_threshold)
    terminate = (u, t, _) -> t > t_min &&
                             DynamicsUtils.get_positions(u)[termination_coord_idx, 1] > thresh

    return run_BO_dynamics(model, atoms, r0_mat, v0, full_data_path;
                           state, tspan = (0.0, austrip(tmax)),
                           dt = austrip(dt), terminate)
end

# -----------------------------------------------------------------------------
# Save paths. Mirrors HonGeAnalysis' dict_to_data_savename but simpler:
# no distributed job IDs, no method key — BO is one method, and the model folder
# is passed explicitly. `savename` is fed a sanitized copy of params so Unitful
# quantities and 1-element vectors collapse to plain numbers for the filename.
# -----------------------------------------------------------------------------

_savename_value(v::Unitful.Quantity) = ustrip(v)
_savename_value(v::AbstractVector{<:Unitful.Quantity}) =
    length(v) == 1 ? ustrip(v[1]) : join(string.(ustrip.(v)), "-")
_savename_value(v::AbstractVector) = length(v) == 1 ? v[1] : v
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
# Parameter sweeps
# -----------------------------------------------------------------------------

all_params_et = Dict{String, Any}(
    "mass"                  => [10.54u"u"],
    "Γ"                     => [0.0u"eV"],
    "r0"                    => [[5.0u"Å"]],              # 1 DOF: bond length
    "translational_kinetic" => [3.0u"eV"],
    "state"                 => [1],
    "tmax"                  => [200.0u"fs"],
    "dt"                    => [0.01u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [1],                       # check r
    "termination_threshold" => [5.0u"Å"],                 # dissociation threshold
)
params_list_et = dict_list(all_params_et)

all_params_pogo = Dict{String, Any}(
    "mass"                  => [(14.007 * 15.999 / (14.007 + 15.999)) * u"u"],
    "Γ"                     => [1.5u"eV"],
    "r0"                    => [[1.15u"Å", 5.0u"Å"]],    # (r, z)
    "translational_kinetic" => [1.0u"eV"],
    "state"                 => [1],
    "tmax"                  => [500.0u"fs"],
    "dt"                    => [0.01u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [2],                       # check z
    "termination_threshold" => [5.0u"Å"],                 # scattered threshold
)
params_list_pogo = dict_list(all_params_pogo)

# -----------------------------------------------------------------------------
# Sweep runner. Skips configs whose .h5 already exists; flip
# `delete_existing_files` to force a re-run.
# -----------------------------------------------------------------------------

delete_existing_files = false

function run_sweep!(runner, params_list, model_folder)
    n = length(params_list)
    for (i, params) in enumerate(params_list)
        savingpath, savingname = dict_to_data_savename(params, model_folder)
        full_data_path = datadir(savingpath, savingname)
        tag = "$i/$n [$model_folder]"
        if isfile(full_data_path)
            if delete_existing_files
                rm(full_data_path)
                @info "$tag deleted existing, re-running" full_data_path
            else
                @info "$tag skipping (already saved)" full_data_path
                continue
            end
        end
        @info "$tag running" full_data_path
        runner(params, full_data_path)
    end
end

run_sweep!(run_erpenbeck_thoss, params_list_et,   "ErpenbeckThoss")
run_sweep!(run_pogo,            params_list_pogo, "NOAu")
