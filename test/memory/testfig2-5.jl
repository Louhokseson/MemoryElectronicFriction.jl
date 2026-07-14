using MemoryElectronicFriction
using Test
using HDF5
using Unitful, UnitfulAtomic
using StaticArrays: SA

# ==============================================================================
# Golden-data regression tests against figure_data/fig_N/fig_N_data.h5.
#
# fig_2 / fig_3: the stored curves are computed by the package itself
# (FrequencyLambda.Lambda from model parameters), so we RECOMPUTE a small
# subsample of ω-points and compare — a true physics regression test.
# Full grids (11001 / 2000 ω-points per curve) are far too slow for CI.
#
# fig_4 / fig_5: the stored curves derive from raw CPA/MD trajectories under
# data/ (gitignored, unavailable on CI), so we can only sanity-check the
# golden files' structure and contents.
# ==============================================================================

const FIGURE_DATA_DIR = joinpath(@__DIR__, "..", "..", "figure_data")

# atomic friction → u⋅ps⁻¹ (matches the plotting scripts' conversion)
const AU_TO_UPS = ustrip(auconvert(u"u", 1) / auconvert(u"ps", 1))

@testset "fig_2 golden data: ErpenbeckThoss Λ(ω) regression" begin
    h5open(joinpath(FIGURE_DATA_DIR, "fig_2", "fig_2_data.h5"), "r") do h5
        Γ_values    = read(h5["Gamma_values"])          # one panel per Γ
        positions   = read(h5["positions"])
        temperature = read(h5["temperature"])
        C_FERMI     = read(h5["C_FERMI"])
        kT_eV       = read(h5["kT_eV"])
        panel_names = ["panel_a", "panel_b", "panel_c"]

        T_au = austrip(temperature * u"K")

        for (k, Γ) in enumerate(Γ_values), pos in positions
            g       = h5[panel_names[k]]["position_$(pos)"]
            ω_ev    = read(g["omega_ev"])
            Λ_ref   = read(g["Lambda"])                 # u⋅ps⁻¹
            ωpk_ref = only(read(g["omega_peak"]))       # eV

            m      = ErpenbeckThossAdsorbate(Γ = austrip(Γ * u"eV"))
            r_au   = austrip(pos * u"Å")

            # subsample: low / mid / high ω
            for idx in (1, length(ω_ev) ÷ 2, length(ω_ev))
                ω_au = austrip(ω_ev[idx] * u"eV")
                Λ    = FrequencyLambda.Lambda(ω_au, m, r_au, T_au) * AU_TO_UPS
                @test isapprox(Λ, Λ_ref[idx]; rtol = 1e-5)
            end

            # peak marker ħω*(x) = |h(x)| + C_FERMI·k_BT
            h_ev = abs(ustrip(auconvert(u"eV",
                       MemoryElectronicFriction.adsorbate_h(r_au, m))))
            @test isapprox(h_ev + C_FERMI * kT_eV, ωpk_ref; rtol = 1e-10)
        end
    end
end

@testset "fig_3 golden data: NOAu 2×2 Λ(ω) regression" begin
    h5open(joinpath(FIGURE_DATA_DIR, "fig_3", "fig_3_data.h5"), "r") do h5
        R_values    = read(h5["R_VALUES"])
        Z_values    = sort(read(h5["Z_VALUES"]))
        temperature = read(h5["TEMPERATURE"])
        ω_ev        = read(h5["ENERGY_GRID"])

        T_au = austrip(temperature * u"K")
        m    = NOAuAdsorbate()

        # one (r, z) per row keeps CI fast; each recomputes 3 full 2×2 tensors
        for r in R_values, z in (Z_values[2],)
            Λ_ref = read(h5["r_$(r)"]["z_$(z)"]["Lambda_mats"])   # Nω × 2 × 2, au

            cfg = SA[austrip(r * u"Å"), austrip(z * u"Å")]
            for idx in (1, length(ω_ev) ÷ 2, length(ω_ev))
                ω_au = austrip(ω_ev[idx] * u"eV")
                Λ    = FrequencyLambda.Lambda(ω_au, m, cfg, T_au)
                @test isapprox(Λ, Λ_ref[idx, :, :]; rtol = 1e-5)
                @test isapprox(Λ, transpose(Λ); rtol = 1e-8)      # symmetry
            end
        end
    end
end

@testset "fig_4 golden data: ΔE(Δ₀) structure" begin
    h5open(joinpath(FIGURE_DATA_DIR, "fig_4", "fig_4_data.h5"), "r") do h5
        Γs  = read(h5["Gamma_values"])
        Δ0s = read(h5["Delta0_values"])
        @test Δ0s ≈ Γs ./ 2
        @test issorted(Γs)

        variants = String[]
        for name in keys(h5["curves"])
            g     = h5["curves"][name]
            ΔE    = read(g["DeltaE_eV"])
            valid = Bool.(read(g["valid"]))
            push!(variants, read(g["variant"]))
            @test length(ΔE) == length(Γs)
            @test any(valid)                       # curve is not empty
            @test all(isfinite, ΔE[valid])         # every valid point is a number
        end
        # the paper figure compares memory vs Markovian CPA
        @test "memory_local" in variants
        @test "markovian" in variants
    end
end

@testset "fig_5 golden data: P(ν) structure" begin
    h5open(joinpath(FIGURE_DATA_DIR, "fig_5", "fig_5_data.h5"), "r") do h5
        ν_inits = read(h5["nu_initial_list"])
        Ets     = read(h5["E_trans_eV_list"])
        ν_max   = read(h5["nu_max"])

        for row in eachindex(ν_inits), col in eachindex(Ets)
            panel = h5["panel_$(row)_$(col)"]
            bins  = read(panel["nu_bins"])
            @test bins == collect(0.0:ν_max)
            for variant in ("memory", "markovian")
                prob = read(panel[variant]["prob"])
                err  = read(panel[variant]["err"])
                @test length(prob) == length(bins)
                @test all(0 .≤ prob .≤ 1)
                @test sum(prob) ≤ 1 + 1e-12        # rejected trajectories may be dropped
                @test all(err .≥ 0)
            end
        end
    end
end

@info "testfig2-5.jl completed"
