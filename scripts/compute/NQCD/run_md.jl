using DrWatson
@quickactivate "HokseonReproduce"

if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end

using NQCModels
using NQCDynamics
using NQCDynamics.InitialConditions: QuantisedDiatomic
using LinearAlgebra: eigen
using Random
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

# 1-DOF slice of a 2-DOF (r, z) BO surface at fixed z. Used by
# QuantisedDiatomic.generate_1D_vibrations to build the bond binding curve:
# the EBK routine expects a ClassicalModel whose `potential` takes a 1x1
# matrix of bond lengths.
struct FrozenHeightBO{M<:NQCModels.ClassicalModels.ClassicalModel} <: NQCModels.ClassicalModels.ClassicalModel
    model2d::M
    z::Float64
end

NQCModels.ndofs(::FrozenHeightBO) = 1

function NQCModels.potential(m::FrozenHeightBO, r::AbstractMatrix)
    return NQCModels.potential(m.model2d, reshape([r[1, 1], m.z], 2, 1))
end

function run_BO_dynamics(bo_model, atoms, dist, full_data_path;
                         tspan        = (0.0, austrip(500u"fs")),
                         dt           = austrip(0.01u"fs"),
                         terminate    = (u, t, _) -> false,
                         trajectories = 1)

    sim = Simulation{Classical}(atoms, bo_model)

    return run_dynamics(sim, tspan, dist; dt,
        callback = DynamicsUtils.TerminatingCallback(terminate),
        output = (OutputPosition, OutputVelocity,
                  OutputKineticEnergy, OutputPotentialEnergy, OutputTotalEnergy),
        trajectories,
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
    dist   = DynamicalDistribution(v0, r0_mat, size(r0_mat))

    quantum_model = NQCModels.ErpenbeckThoss(; Γ = austrip(Γ))
    bo_model      = BOAdiabaticModel(quantum_model, state)

    t_min  = austrip(termination_min_time)
    thresh = austrip(termination_threshold)
    terminate = (u, t, _) -> t > t_min &&
                             DynamicsUtils.get_positions(u)[termination_coord_idx, 1] > thresh

    return run_BO_dynamics(bo_model, atoms, dist, full_data_path;
                           tspan = (0.0, austrip(tmax)),
                           dt = austrip(dt), terminate)
end

# Translation is deterministic: fixed z0, fixed ż chosen so NQCDynamics'
# KE(z-DOF) = 0.5 * atoms.masses[1] * ż² matches `translational_kinetic`.
# (POGO is a 1-atom/2-DOF model so atoms.masses has length 1 and the same
# mass backs both the bond and the z-DOF in KE bookkeeping.)
# The bond DOF is either frozen at r0[1] with zero radial velocity
# (vibrational_state === nothing) or EBK-sampled at quantum number ν
# (Integer). Seed keyed off (ν, E_trans, N) so equivalent configs reproduce.
function noau_initial_distribution(bo_model, atoms, params; trajectories=1)
    @unpack r0, translational_kinetic, vibrational_state = params

    r_bond0 = austrip(r0[1])
    z0      = austrip(r0[2])
    μ_bond  = atoms.masses[1]        # bond reduced mass; also drives z-DOF KE
    ż       = -sqrt(2 * austrip(translational_kinetic) / μ_bond)

    if vibrational_state === nothing
        v_samples = [reshape([0.0, ż],      2, 1) for _ in 1:trajectories]
        r_samples = [reshape([r_bond0, z0], 2, 1) for _ in 1:trajectories]
    else
        ν = Int(vibrational_state)
        Random.seed!(hash((ν, austrip(translational_kinetic), trajectories)))
        model1d = FrozenHeightBO(bo_model, z0)
        bonds, bond_vs = QuantisedDiatomic.generate_1D_vibrations(
            model1d, μ_bond, ν; samples=trajectories)
        v_samples = [reshape([bond_vs[k], ż], 2, 1) for k in 1:trajectories]
        r_samples = [reshape([bonds[k],   z0], 2, 1) for k in 1:trajectories]
    end

    return DynamicalDistribution(v_samples, r_samples, (2, 1))
end

function run_NOAu(params, full_data_path)
    @unpack mass, r0, translational_kinetic, state, tmax, dt,
            termination_min_time, termination_coord_idx, termination_threshold,
            trajectories = params

    atoms         = Atoms(mass)
    quantum_model = POGOModel()
    bo_model      = BOAdiabaticModel(quantum_model, state)
    dist          = noau_initial_distribution(bo_model, atoms, params; trajectories)

    t_min  = austrip(termination_min_time)
    thresh = austrip(termination_threshold)
    terminate = (u, t, _) -> t > t_min &&
                             DynamicsUtils.get_positions(u)[termination_coord_idx, 1] > thresh

    return run_BO_dynamics(bo_model, atoms, dist, full_data_path;
                           tspan = (0.0, austrip(tmax)),
                           dt = austrip(dt), terminate, trajectories)
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
# Parameter sweeps
# -----------------------------------------------------------------------------

all_params_et = Dict{String, Any}(
    "mass"                  => [10.54u"u"],
    "Γ"                     => [0.25u"eV"],
    "r0"                    => [[5.0u"Å"]],              # 1 DOF: surface distance
    "translational_kinetic" => [3.0u"eV"],
    "state"                 => [1],
    "tmax"                  => [200.0u"fs"],
    "dt"                    => [0.01u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [1],                       # check r
    "termination_threshold" => [5.0u"Å"],                 # dissociation threshold
)
params_list_et = dict_list(all_params_et)

all_params_NOAu = Dict{String, Any}(
    "mass"                  => [(14.007 * 15.999 / (14.007 + 15.999)) * u"u"],   # μ_NO — POGO is 1-atom
#    "Γ"                     => [1.5u"eV"], ## constant 1.5 eV
    "r0"                    => [[1.15u"Å", 5.0u"Å"]],    # (r, z); r0[1] is the frozen bond length when vibrational_state=nothing
    "translational_kinetic" => [1.0u"eV"],
    "state"                 => [1],
    "tmax"                  => [500.0u"fs"],
    "dt"                    => [0.25u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [2],                       # check z
    "termination_threshold" => [5.0u"Å"],                 # scattered threshold
    # nothing → frozen bond (old behaviour). Integer ν → EBK-sample (r, ṙ)
    # at quantum number ν; bump trajectories to ~1000 for a ν ensemble.
    "vibrational_state"     => [nothing],                 # try [nothing, 0, 3, 16]
    "trajectories"          => [1],
)
params_list_NOAu = dict_list(all_params_NOAu)

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
run_sweep!(run_NOAu,            params_list_NOAu, "NOAu")