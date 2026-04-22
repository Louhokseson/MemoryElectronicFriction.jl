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
using CairoMakie
using HokseonPlots

# -----------------------------------------------------------------------------
# Born-Oppenheimer classical MD on a NQCModels QuantumModel.
#
# A `Simulation{Classical}` needs a scalar potential, so we lift any QuantumModel
# onto a single adiabatic surface by diagonalising V(R) and selecting one
# eigenvalue. `state = 1` is the electronic ground state.
#
# We define a small local wrapper rather than using NQCModels.AdiabaticStateSelector:
# the library version calls `NQCModels.derivative(quantum_model, r)`, but
# ErpenbeckThoss (v1.0.1) overrides that allocating method to return a bare
# 2x2 Hermitian instead of the (ndofs x natoms) Matrix{Hermitian} the interface
# requires, which crashes AdiabaticStateSelector. Going through
# `zero_derivative` + `derivative!` avoids the broken override.
# -----------------------------------------------------------------------------

struct BOAdiabaticModel{M<:NQCModels.QuantumModels.QuantumModel} <: NQCModels.ClassicalModels.ClassicalModel
    quantum_model::M
    state::Int
end

NQCModels.ndofs(m::BOAdiabaticModel) = NQCModels.ndofs(m.quantum_model)

function NQCModels.potential(m::BOAdiabaticModel, r::AbstractMatrix)
    V = NQCModels.potential(m.quantum_model, r)
    return eigen(V).values[m.state]
end

function NQCModels.derivative!(m::BOAdiabaticModel, output::AbstractMatrix, r::AbstractMatrix)
    V = NQCModels.potential(m.quantum_model, r)
    U = eigen(V).vectors
    D = NQCModels.zero_derivative(m.quantum_model, r)
    NQCModels.derivative!(m.quantum_model, D, r)
    for I in eachindex(output, D)
        output[I] = (U' * D[I] * U)[m.state, m.state]
    end
    return output
end

function run_BO_dynamics(quantum_model, atoms, r0, v0;
                         state::Int = 1,
                         tspan      = (0.0, austrip(500u"fs")),
                         dt         = austrip(0.1u"fs"))

    bo_model = BOAdiabaticModel(quantum_model, state)
    sim      = Simulation(atoms, bo_model)

    u0 = DynamicsVariables(sim, v0, r0)

    return run_dynamics(sim, tspan, u0; dt = dt,
        callback = terminate,
        output = (OutputPosition, OutputVelocity,
                  OutputKineticEnergy, OutputPotentialEnergy, OutputTotalEnergy))
end

function termination_condition(u, t, integrator)::Bool
    # make sure the dynamics doesn't stop at the very beginning
    return (t > austrip(10u"fs")) && (DynamicsUtils.get_positions(u)[1] > austrip(5u"Å"))
end
terminate = DynamicsUtils.TerminatingCallback(termination_condition)

# -----------------------------------------------------------------------------
# 1. ErpenbeckThoss (1 DOF, 2 electronic states)
# -----------------------------------------------------------------------------

function run_erpenbeck_thoss()
    model = NQCModels.ErpenbeckThoss(Γ = austrip(0.06u"eV"))
    atoms = Atoms(1u"u")

    # Start slightly displaced from the Morse minimum (x₀ = 1.78 Å) at rest.
    r0 = fill(austrip(2.0u"Å"), 1, 1)
    v0 = zeros(1, 1)

    return run_BO_dynamics(model, atoms, r0, v0;
                           tspan = (0.0, austrip(200u"fs")),
                           dt    = austrip(0.1u"fs"))
end

# -----------------------------------------------------------------------------
# 2. POGOModel (2 DOFs: r = N-O bond, z = molecule-surface, 2 electronic states)
# -----------------------------------------------------------------------------

function run_pogo()
    model = POGOModel()

    # POGOModel treats the adsorbate as a single "particle" with 2 DOFs.
    # Use the NO reduced mass as the effective mass for both coordinates
    # (the same choice made elsewhere in this project — see plot_Lambda_NOAu.jl).
    μ_NO  = austrip((14.007 * 15.999 / (14.007 + 15.999)) * u"u")
    atoms = Atoms([μ_NO])

    # r ≈ N-O equilibrium (~1.15 Å), z well above the surface
    r0 = reshape([austrip(1.15u"Å"), austrip(5.0u"Å")], 2, 1)
    v0 = zeros(2, 1)

    # Give z a small incoming velocity (toward the surface) to make the
    # trajectory non-trivial.
    v0[2, 1] = -austrip(0.002u"Å/fs")

    return run_BO_dynamics(model, atoms, r0, v0;
                           tspan = (0.0, austrip(500u"fs")),
                           dt    = austrip(0.1u"fs"))
end

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

function plot_trajectories(traj_et, traj_pogo)
    t_et = ustrip.(auconvert.(u"fs", traj_et[:Time]))
    r_et = [ustrip(auconvert(u"Å", R[1, 1])) for R in traj_et[:OutputPosition]]
    E_et = ustrip.(auconvert.(u"eV", traj_et[:OutputTotalEnergy]))

    t_pg = ustrip.(auconvert.(u"fs", traj_pogo[:Time]))
    r_pg = [ustrip(auconvert(u"Å", R[1, 1])) for R in traj_pogo[:OutputPosition]]
    z_pg = [ustrip(auconvert(u"Å", R[2, 1])) for R in traj_pogo[:OutputPosition]]
    E_pg = ustrip.(auconvert.(u"eV", traj_pogo[:OutputTotalEnergy]))

    fig = Figure(size = (HokseonPlots.RESOLUTION[1]*3, 3*HokseonPlots.RESOLUTION[2]))

    ax1 = Axis(fig[1, 1]; xlabel = "t / fs", ylabel = "r / Å",
                 title = "ErpenbeckThoss BO-MD")
    lines!(ax1, t_et, r_et; linewidth = 2)

    ax2 = Axis(fig[1, 2]; xlabel = "t / fs", ylabel = "E_total / eV",
                 title = "ErpenbeckThoss energy")
    lines!(ax2, t_et, E_et; linewidth = 2)

    ax3 = Axis(fig[2, 1]; xlabel = "t / fs", ylabel = "coordinate / Å",
                 title = "POGOModel BO-MD")
    lines!(ax3, t_pg, r_pg; linewidth = 2, label = "r (N–O)")
    lines!(ax3, t_pg, z_pg; linewidth = 2, label = "z (surface)")
    axislegend(ax3; position = :rt)

    ax4 = Axis(fig[2, 2]; xlabel = "t / fs", ylabel = "E_total / eV",
                 title = "POGOModel energy")
    lines!(ax4, t_pg, E_pg; linewidth = 2)

    return fig
end

traj_et   = run_erpenbeck_thoss()
traj_pogo = run_pogo()
display(plot_trajectories(traj_et, traj_pogo))
