using MemoryElectronicFriction
using Test
using HDF5
using Distributed
using Unitful, UnitfulAtomic

# ==============================================================================
# Reduced-scale golden regression test of the CPA ΔE engine
# (scripts/compute/cpa/DeltaE.jl) — the code path behind fig_4/fig_5.
#
# The paper-scale computation (20k trajectory positions × 2011-point ω-grid,
# per Γ) is HPC-only, and its MD inputs live in the gitignored data/ tree, so
# CI cannot reproduce figure_data/fig_4/fig_5 exactly. Instead we run the SAME
# delta_energy pipeline — Λ(ω) → cosine transform → kernel double sum, plus
# the Markovian single sum — on a small synthetic ErpenbeckThoss trajectory
# and compare against a committed golden reference.
#
# Regenerate the golden file after an INTENDED physics change with:
#   GENERATE_GOLDEN=1 julia --project=. -e 'include("test/memory/testDeltaE.jl")'
# ==============================================================================

include(joinpath(@__DIR__, "..", "..", "scripts", "compute", "cpa", "DeltaE.jl"))

const GOLDEN_PATH = joinpath(@__DIR__, "..", "golden", "cpa_DeltaE_reduced.h5")
const GENERATE_GOLDEN = get(ENV, "GENERATE_GOLDEN", "0") == "1"

# --- reduced, fully deterministic inputs --------------------------------------
const ΔE_TEST_Γ_eV  = 0.2
const ΔE_TEST_T_K   = 300
# coarse variant of DEFAULT_ω_GRID_eV (same structure: DC point, log crossover,
# linear tail)
const ΔE_TEST_ω_eV  = vcat(0.0,
                           [1e-7, 3.16e-7, 1e-6, 3.16e-6, 1e-5, 3.16e-5,
                            1e-4, 3.16e-4, 1e-3, 3.16e-3],
                           collect(0.1:0.1:20.0))

# Synthetic 1-DOF bond trajectory: oscillation around 1.95 Å — smooth, analytic,
# platform-independent. 25 steps of 2 fs.
function synthetic_et_trajectory()
    N     = 25
    dt_au = austrip(2.0u"fs")
    t     = collect(0:N-1) .* dt_au
    r_c   = austrip(1.95u"Å")
    A     = austrip(0.15u"Å")
    Ω     = 2π / austrip(30.0u"fs")
    Q     = reshape(r_c .+ A .* cos.(Ω .* t), 1, N)
    V     = reshape(-A .* Ω .* sin.(Ω .* t), 1, N)
    return (t = t, OutputPosition = Q, OutputVelocity = V), dt_au
end

@testset "fig_4 pipeline: reduced CPA ΔE golden regression" begin
    md_params = Dict{String, Any}("Γ" => ΔE_TEST_Γ_eV * u"eV", "mass" => [10.54u"u"])
    adsorbate = build_adsorbate(Val(:ErpenbeckThoss), md_params)
    traj, dt_au = synthetic_et_trajectory()
    ω_au = austrip.(ΔE_TEST_ω_eV .* u"eV")
    T_au = austrip(ΔE_TEST_T_K * u"K")

    # memory CPA — both kernel-averaging schemes used by the fig_4 scripts
    ΔE_endpoint = only(delta_energy(:ErpenbeckThoss, adsorbate, traj, dt_au;
                                    ω_au, T_au, kernel_average = :endpoint,
                                    progress = false))
    ΔE_arith    = only(delta_energy(:ErpenbeckThoss, adsorbate, traj, dt_au;
                                    ω_au, T_au, kernel_average = :arithmetic,
                                    progress = false))

    # Markovian CPA — NQCD wide-band-exact friction along the same trajectory
    sim   = build_friction_sim(Val(:ErpenbeckThoss), md_params; T_au)
    ΔE_mark = only(delta_energy(:ErpenbeckThoss, sim, traj, dt_au;
                                progress = false))

    @test isfinite(ΔE_endpoint)
    @test isfinite(ΔE_arith)
    @test isfinite(ΔE_mark)
    @test ΔE_mark > 0            # friction dissipates energy

    if GENERATE_GOLDEN
        mkpath(dirname(GOLDEN_PATH))
        h5open(GOLDEN_PATH, "w") do h5
            h5["Gamma_eV"]        = ΔE_TEST_Γ_eV
            h5["T_K"]             = ΔE_TEST_T_K
            h5["omega_grid_eV"]   = ΔE_TEST_ω_eV
            h5["DeltaE_endpoint"] = ΔE_endpoint
            h5["DeltaE_arith"]    = ΔE_arith
            h5["DeltaE_markovian"] = ΔE_mark
        end
        @info "Golden reference written" GOLDEN_PATH ΔE_endpoint ΔE_arith ΔE_mark
    else
        @assert isfile(GOLDEN_PATH) "golden file missing — run with GENERATE_GOLDEN=1"
        h5open(GOLDEN_PATH, "r") do h5
            @test read(h5["Gamma_eV"]) == ΔE_TEST_Γ_eV        # params in sync
            @test read(h5["omega_grid_eV"]) ≈ ΔE_TEST_ω_eV
            @test isapprox(ΔE_endpoint, read(h5["DeltaE_endpoint"]);  rtol = 1e-6)
            @test isapprox(ΔE_arith,    read(h5["DeltaE_arith"]);     rtol = 1e-6)
            @test isapprox(ΔE_mark,     read(h5["DeltaE_markovian"]); rtol = 1e-6)
        end
    end
end

@info "testDeltaE.jl completed"
