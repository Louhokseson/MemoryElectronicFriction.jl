using DrWatson
@quickactivate "MemoryElectronicFriction"

using MemoryElectronicFriction
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

# NQCDynamics gives every DOF of an atom that atom's mass, so POGO's 2-DOF ×
# 1-atom layout puts μ on both r and z. Presenting (r, z) as two 1-DOF "atoms"
# (positions 1×2) lets atom 1 = bond length r carry the reduced mass μ and
# atom 2 = centre-of-mass height z carry the total mass M = m_N + m_O,
# i.e. mass matrix diag(μ, M).
struct RZAsAtoms{M<:NQCModels.ClassicalModels.ClassicalModel} <: NQCModels.ClassicalModels.ClassicalModel
    model::M
end

NQCModels.ndofs(::RZAsAtoms) = 1

function NQCModels.potential(m::RZAsAtoms, R::AbstractMatrix)
    return NQCModels.potential(m.model, reshape(R, 2, 1))
end

function NQCModels.derivative!(m::RZAsAtoms, D::AbstractMatrix, R::AbstractMatrix)
    D2 = zeros(2, 1)
    NQCModels.derivative!(m.model, D2, reshape(R, 2, 1))
    D[1, 1], D[1, 2] = D2[1, 1], D2[2, 1]
    return D
end

function run_BO_dynamics(bo_model, atoms, dist, full_data_path;
                         tspan        = (0.0, austrip(500u"fs")),
                         dt           = austrip(0.01u"fs"),
                         terminate    = (u, t, _) -> false,
                         trajectories = 1,
                         selection    = nothing)

    sim = Simulation{Classical}(atoms, bo_model)

    return run_dynamics(sim, tspan, dist; dt,
        callback = DynamicsUtils.TerminatingCallback(terminate),
        output = (OutputPosition, OutputVelocity,
                  OutputKineticEnergy, OutputPotentialEnergy, OutputTotalEnergy),
        trajectories, selection,
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

# Translation is deterministic: fixed z0, fixed ż chosen so the centre-of-mass
# KE 0.5 * M * ż² (M = atoms.masses[2]) matches `translational_kinetic`.
# Positions/velocities are 1×2 [r z] for RZAsAtoms.
# The bond DOF is either frozen at r0[1] with zero radial velocity
# (vibrational_state === nothing) or EBK-sampled at quantum number ν
# (Integer) with the reduced mass μ = atoms.masses[1]. `bo_model` is the
# plain 2×1 (r, z) model, as FrozenHeightBO expects.
# Seed keyed off (ν, E_trans, N) so equivalent configs reproduce.
function noau_initial_distribution(bo_model, atoms, params; trajectories=1)
    @unpack r0, translational_kinetic, vibrational_state = params

    r_bond0 = austrip(r0[1])
    z0      = austrip(r0[2])
    μ_bond  = atoms.masses[1]        # bond reduced mass
    M_tot   = atoms.masses[2]        # total mass, carried by the centre-of-mass height z
    ż       = -sqrt(2 * austrip(translational_kinetic) / M_tot)

    if vibrational_state === nothing
        v_samples = [[0.0     ż ] for _ in 1:trajectories]
        r_samples = [[r_bond0 z0] for _ in 1:trajectories]
    else
        ν = Int(vibrational_state)
        Random.seed!(hash((ν, austrip(translational_kinetic), trajectories)))
        model1d = FrozenHeightBO(bo_model, z0)
        bonds, bond_vs = QuantisedDiatomic.generate_1D_vibrations(
            model1d, μ_bond, ν; samples=trajectories)
        v_samples = [[bond_vs[k] ż ] for k in 1:trajectories]
        r_samples = [[bonds[k]   z0] for k in 1:trajectories]
    end

    return DynamicalDistribution(v_samples, r_samples, (1, 2))
end

function run_NOAu(params, full_data_path)
    @unpack mass, r0, translational_kinetic, state, tmax, dt,
            termination_min_time, termination_coord_idx, termination_threshold,
            trajectories = params

    atoms         = Atoms(austrip.(mass))   # [μ, M]
    quantum_model = POGOModel()
    bo_model      = BOAdiabaticModel(quantum_model, state)
    dist          = noau_initial_distribution(bo_model, atoms, params; trajectories)

    t_min  = austrip(termination_min_time)
    thresh = austrip(termination_threshold)
    terminate = (u, t, _) -> t > t_min &&
                             DynamicsUtils.get_positions(u)[1, termination_coord_idx] > thresh

    # selection = 1:N runs each sampled initial condition exactly once; the
    # NQCDynamics default (nothing) draws from `dist` with replacement.
    return run_BO_dynamics(RZAsAtoms(bo_model), atoms, dist, full_data_path;
                           tspan = (0.0, austrip(tmax)),
                           dt = austrip(dt), terminate, trajectories,
                           selection = 1:trajectories)
end

# Save paths (`dict_to_data_savename`) and the trajectory loader are part of
# MemoryElectronicFriction.IO and re-exported from the top-level module.

# -----------------------------------------------------------------------------
# Parameter sweeps
# -----------------------------------------------------------------------------

all_params_et = Dict{String, Any}(
    "mass"                  => [10.54u"u"],
    "Γ"                     => [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0] .* u"eV",
    "r0"                    => [[5.0u"Å"]],              # 1 DOF: surface distance
    "translational_kinetic" => [0.5, 1.0, 2.0, 3.0].*u"eV",
    "state"                 => [1],
    "tmax"                  => [200.0u"fs"],
    "dt"                    => [0.01u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [1],                       # check r
    "termination_threshold" => [5.0u"Å"],                 # dissociation threshold
)
params_list_et = dict_list(all_params_et)

all_params_NOAu = Dict{String, Any}(
    "mass"                  => [[14.007 * 15.999 / (14.007 + 15.999), 14.007 + 15.999] .* u"u"],   # [μ_NO, M_NO]: r carries μ, z carries M
#    "Γ"                     => [1.5u"eV"], ## constant 1.5 eV
    "r0"                    => [[1.15u"Å", 5.0u"Å"]],    # (r, z); r0[1] is the frozen bond length when vibrational_state=nothing
    "translational_kinetic" => collect(0.2:0.1:1.0) .* u"eV",
    "state"                 => [1],
    "tmax"                  => [1000.0u"fs"],
    "dt"                    => [0.25u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [2],                       # check z
    "termination_threshold" => [5.0u"Å"],                 # scattered threshold
    # nothing → frozen bond (old behaviour). Integer ν → EBK-sample (r, ṙ)
    # at quantum number ν; bump trajectories to ~1000 for a ν ensemble.
    "vibrational_state"     => [0, 3, 16],                 # try [nothing, 0, 3, 16]
    "trajectories"          => [1000],
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