using DrWatson
@quickactivate "MemoryElectronicFriction"

using HDF5
using LinearAlgebra: eigen
using Unitful, UnitfulAtomic
using CairoMakie

using MemoryElectronicFriction
using HokseonPlots
using HokseonAssistant
using QuadGK: quadgk
using Roots: find_zero

using NQCModels
# Sibling of `plot_vib_state_distribution_paper.jl` — same data (memory CPA
# with `kernel_average=:endpoint` vs Markovian CPA at T = 300 K, diag(μ, M)
# MD), same layout. The only difference is what we do with the continuous EBK
# quantum numbers ν_c: instead of rounding them onto integer states, we put a
# Gaussian of width σ (our choice) at each ν_c, sum them and normalise,
#
#     ρ(ν) = (1 / (N σ √(2π))) Σ_i exp(−½ ((ν − ν_c[i]) / σ)²),
#
# so ∫ρ dν = 1 and ∫_{n−½}^{n+½} ρ(ν) dν ≈ P_n of the paper figure.
#
# Trajectories that friction pushes below the bottom of the well (E < V_min)
# have no turning points, so EBK gives them no ν_c and the paper histogram
# drops them. Here we instead continue ν_c linearly in energy,
#     ν_c = −½ + (E − V_min) / ħω₀,
# which is the EBK result's own E → V_min limit and joins it smoothly at
# ν = −½. The shaded region ν < −½ marks these below-well states.

# ---------------------------------------------------------------------------
# 1D adiabatic surface for the NO bond — identical to the paper script.
# ---------------------------------------------------------------------------

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

struct FrozenHeightBO{M<:NQCModels.ClassicalModels.ClassicalModel} <: NQCModels.ClassicalModels.ClassicalModel
    model2d::M
    z::Float64
end

NQCModels.ndofs(::FrozenHeightBO) = 1

function NQCModels.potential(m::FrozenHeightBO, r::AbstractMatrix)
    return NQCModels.potential(m.model2d, reshape([r[1, 1], m.z], 2, 1))
end

struct DirectPotential{F, T<:Real}
    V1              :: F   # callable V₁(r::Real)::Real (atomic units)
    μ               :: T
    r_eq            :: T
    V_min           :: T
    ω0              :: T   # harmonic frequency at r_eq (ħ = 1)
    V_outer         :: T   # V₁ evaluated at r_outer_bracket — outer wall reference
    r_inner_bracket :: T
    r_outer_bracket :: T
end

(V::DirectPotential)(r) = V.V1(r)

function build_direct_potential(; z_eval_Å::Real = 10.0,
                                  r_inner_Å::Real = 0.5,
                                  r_outer_Å::Real = 5.0,
                                  r_eq_grid_Å      = 0.7:0.005:2.0)
    μ_NO  = austrip((14.007 * 15.999 / (14.007 + 15.999)) * u"u")
    bo_2d = BOAdiabaticModel(POGOModel(), 1)
    bo_1d = FrozenHeightBO(bo_2d, austrip(z_eval_Å * u"Å"))

    V1(r) = NQCModels.potential(bo_1d, [r;;])

    rs_au   = austrip.(r_eq_grid_Å .* u"Å")
    Vs      = V1.(rs_au)
    idx_min = argmin(Vs)
    @assert 1 < idx_min < length(rs_au) "well minimum at the r_eq_grid_Å boundary — extend the grid"
    h       = 1e-5
    dV(r)   = (V1(r + h) - V1(r - h)) / (2h)
    r_eq    = find_zero(dV, (rs_au[idx_min - 1], rs_au[idx_min + 1]))
    V_min   = V1(r_eq)
    h2      = 1e-4
    ω0      = sqrt((V1(r_eq + h2) - 2V_min + V1(r_eq - h2)) / h2^2 / μ_NO)

    r_inner_au = austrip(r_inner_Å * u"Å")
    r_outer_au = austrip(r_outer_Å * u"Å")
    V_outer    = V1(r_outer_au)

    return DirectPotential(V1, μ_NO, r_eq, V_min, ω0, V_outer, r_inner_au, r_outer_au)
end

# ---------------------------------------------------------------------------
# EBK quantisation from a given total bond energy — as in the paper script,
# except below the well, where we return the harmonic continuation instead of
# a sentinel (see header).
# ---------------------------------------------------------------------------

const _NU_DISSOCIATED = Inf   # E above the outer-wall reference

function quantise_from_energy(E::Real, V::DirectPotential)
    E ≤ V.V_min   && return (E - V.V_min) / V.ω0 - 1 / 2
    E ≥ V.V_outer && return _NU_DISSOCIATED
    r₁ = find_zero(r -> V.V1(r) - E, (V.r_inner_bracket, V.r_eq))
    r₂ = find_zero(r -> V.V1(r) - E, (V.r_eq,            V.r_outer_bracket))
    integral, _ = quadgk(r -> sqrt(max(E - V.V1(r), 0.0)), r₁, r₂; maxevals = 200)
    return sqrt(2 * V.μ) / π * integral - 1 / 2
end

function final_vib_states(params_md::Dict{String,Any}, cpa_cfg::Dict{String,Any},
                          V::DirectPotential)
    md_savingpath, md_savingname = dict_to_data_savename(params_md, "NOAu")
    md_path = datadir(md_savingpath, md_savingname)
    isfile(md_path) || (@warn "Missing MD file" md_path; return nothing)
    trajs = load_md_trajectories(md_path)

    cpa_savingpath, cpa_savingname = CPA_dict_to_data_savename(params_md, cpa_cfg)
    cpa_path = datadir(cpa_savingpath, cpa_savingname)
    isfile(cpa_path) || (@warn "Missing CPA file" cpa_path; return nothing)
    ΔE_au_mat = h5open(cpa_path, "r") do f
        read(f, "DeltaE_au")
    end
    if !(ndims(ΔE_au_mat) == 2 && size(ΔE_au_mat, 1) == 2)
        @warn "DeltaE_au is not a 2 × n_traj matrix — skipping" cpa_path size=size(ΔE_au_mat)
        return nothing
    end
    ΔE_r = @view ΔE_au_mat[1, :]
    n = min(length(trajs), length(ΔE_r))

    ν_cpa = Vector{Float64}(undef, n)
    for i in 1:n
        r_f = trajs[i].OutputPosition[1, end]
        v_f = trajs[i].OutputVelocity[1, end]
        E_bond = 0.5 * V.μ * v_f^2 + V(r_f)       # J=0 → V(r) = V_1d(r)
        ν_cpa[i] = quantise_from_energy(E_bond - ΔE_r[i], V)
    end
    return ν_cpa
end

# ---------------------------------------------------------------------------
# Sum of Gaussians of width σ centred at each ν_c, normalised to unit area.
# A dissociated ν_c = Inf adds exp(−Inf) = 0 but still counts in N.
# ---------------------------------------------------------------------------

function vib_kde(ν_continuous::AbstractVector{<:Real}, ν_grid::AbstractVector{<:Real}; σ::Real)
    ρ = zeros(Float64, length(ν_grid))
    for νc in ν_continuous
        @. ρ += exp(-0.5 * ((ν_grid - νc) / σ)^2)
    end
    return ρ ./ (length(ν_continuous) * σ * sqrt(2π))
end

kernel_avg_of(cfg) = get(cfg, "kernel_average", nothing)
is_markovian(cfg)  = !haskey(cfg, "kernel_average")

# ---------------------------------------------------------------------------
# MD params builder — must match the paper script exactly, otherwise the
# savename points at a different MD file.
# ---------------------------------------------------------------------------

function build_md_params(E_TRANS_eV::Real, ν_initial::Integer)
    return Dict{String, Any}(
        "mass"                  => [14.007 * 15.999 / (14.007 + 15.999), 14.007 + 15.999] .* u"u",   # [μ_NO, M_NO]: r carries μ, z carries M
        "r0"                    => [1.15u"Å", 5.0u"Å"],
        "translational_kinetic" => E_TRANS_eV * u"eV",
        "state"                 => 1,
        "tmax"                  => 1000.0u"fs",
        "dt"                    => 0.25u"fs",
        "termination_min_time"  => 10.0u"fs",
        "termination_coord_idx" => 2,
        "termination_threshold" => 5.0u"Å",
        "vibrational_state"     => ν_initial,
        "trajectories"          => 1000,
    )
end

# ---------------------------------------------------------------------------
# Figure layout: rows = ν_initial, cols = Eₜ — same grid, fonts and colours as
# the paper figure, with KDE curves in place of the histograms.
# ---------------------------------------------------------------------------

function plot_vib_state_distribution_kde(ν_INITIAL_LIST, E_TRANS_eV_LIST,
                                         configs::AbstractVector;
                                         T_focus          = 300,
                                         σ                = 0.1,
                                         n_grid           = 800,
                                         figsize          = (HokseonPlots.TWO_COLUMN_WIDTH,
                                                             HokseonPlots.TWO_COLUMN_WIDTH),
                                         panel_spinewidth = 1.8,
                                         ylims            = (0.0, nothing),
                                         title_fontsize   = 20,
                                         label_fontsize   = 20,
                                         tick_fontsize    = 18,
                                         legend_fontsize  = 18,
                                         panel_label_fontsize = 18)

    V = build_direct_potential()
    @info "Built direct 1D potential" V_min_eV=ustrip(auconvert(u"eV", V.V_min)) r_eq_Å=ustrip(auconvert(u"Å", V.r_eq)) V_outer_eV=ustrip(auconvert(u"eV", V.V_outer))

    endpt_cfg = nothing
    mark_cfg  = nothing
    for c in configs
        c["T_K"] == T_focus || continue
        if is_markovian(c)
            mark_cfg === nothing && (mark_cfg = c)
        elseif kernel_avg_of(c) == :endpoint && endpt_cfg === nothing
            endpt_cfg = c
        end
    end
    endpt_cfg === nothing && error("No endpoint-kernel memory CPA config at T = $(T_focus) K in `configs`")
    mark_cfg  === nothing && error("No Markovian CPA config at T = $(T_focus) K in `configs`")

    n_rows = length(ν_INITIAL_LIST)
    n_cols = length(E_TRANS_eV_LIST)

    color_endpt = "#B53A8C"
    color_mark  = "#3F9DCC"

    fig = Figure(; size = figsize,
                   figure_padding = (8, 12, 6, 6),
                   fonts = (; regular = projectdir("fonts", "MinionPro-Capt.otf")))

    for (c, Eₜ) in enumerate(E_TRANS_eV_LIST)
        Label(fig[1, c], "Eₜ = $(Eₜ) eV";
              fontsize   = title_fontsize,
              halign     = :center,
              tellwidth  = false,
              tellheight = true,
              padding    = (0, 0, 2, 0))
    end

    function row_xspec(ν_init)
        if ν_init ≤ 5
            # Starts at −2.5 so the below-well continuation (ν_c down to
            # ≈ −1.9 for memory CPA at Eₜ = 1 eV) is fully on the axis. Ticks
            # every 2 so "4" and the next panel's "−2" don't collide at colgap 0.
            return (xticks = -2:2:4,        xlims = (-2.5, 5.5))
        elseif ν_init ≤ 10
            return (xticks = 0:2:10,        xlims = (-0.5, 10.5))
        else
            return (xticks = 0:5:15,        xlims = (-1, 17))
        end
    end

    axes = Matrix{Any}(undef, n_rows, n_cols)
    for (r_idx, ν_init) in enumerate(ν_INITIAL_LIST)
        spec   = row_xspec(ν_init)
        ν_grid = range(spec.xlims[1], spec.xlims[2]; length = n_grid)
        for (c_idx, Eₜ) in enumerate(E_TRANS_eV_LIST)
            ax = Axis(fig[r_idx + 1, c_idx];
                      xticks = spec.xticks,
                      xticklabelsize = tick_fontsize,
                      yticklabelsize = tick_fontsize,
                      xtickalign = 1.0,
                      ytickalign = 1.0,
                      xticksmirrored = true,
                      yticksmirrored = true,
                      limits = (spec.xlims[1], spec.xlims[2], ylims[1], ylims[2]),
                      yautolimitmargin = (0.05, 0.15),   # headroom for the panel letter
                      spinewidth = panel_spinewidth,
                      xtickwidth = panel_spinewidth,
                      ytickwidth = panel_spinewidth,
                      xgridvisible = false, ygridvisible = false,
                      xminorticksvisible = false, yminorticksvisible = false)

            params_md = build_md_params(Eₜ, ν_init)
            ν_endpt   = final_vib_states(params_md, endpt_cfg, V)
            ν_mark    = final_vib_states(params_md, mark_cfg,  V)

            vspan!(ax, spec.xlims[1], -0.5; color = (:grey, 0.15))
            vlines!(ax, [ν_init]; color = :black, linestyle = :dash,
                    linewidth = 1.8)

            for (νs, col) in ((ν_endpt, color_endpt), (ν_mark, color_mark))
                νs === nothing && continue
                lines!(ax, ν_grid, vib_kde(νs, ν_grid; σ = σ);
                       color = col, linewidth = 2.0)
            end

            panel_letter = Char('a' + (r_idx - 1) * n_cols + (c_idx - 1))
            text!(ax, 0.04, 0.95;
                  text        = "($(panel_letter))",
                  space       = :relative,
                  align       = (:left, :top),
                  fontsize    = panel_label_fontsize,
                  color       = :black,
                  strokewidth = 0.5,
                  strokecolor = :black)

            axes[r_idx, c_idx] = ax
        end
    end

    for r_idx in 1:n_rows
        linkyaxes!(axes[r_idx, :]...)
    end
    for r_idx in 1:n_rows, c_idx in 2:n_cols
        hideydecorations!(axes[r_idx, c_idx]; grid = false, ticks = false)
    end

    Label(fig[:, 0], "Probability density";
          fontsize = label_fontsize,
          rotation = π / 2,
          tellheight = false,
          padding = (0, 0, 0, 0))
    Label(fig[n_rows + 2, 1:n_cols], "Final vibrational state ν";
          fontsize = label_fontsize,
          tellwidth = false,
          padding = (0, 0, 0, 0))

    legend_elems  = [
        LineElement(; color = color_endpt, linewidth = 2.0),
        LineElement(; color = color_mark,  linewidth = 2.0),
        LineElement(; color = :black, linestyle = :dash, linewidth = 1.8),
    ]
    legend_labels = ["Memory", "Markovian", "Initial vibrational state"]

    Legend(fig[n_rows + 3, 1:n_cols], legend_elems, legend_labels;
           orientation = :horizontal,
           nbanks = 1,
           tellwidth = false,
           tellheight = true,
           framevisible = false,
           labelsize = legend_fontsize,
           patchsize = (24, 12),
           margin = (0, 0, 0, 0))

    rowgap!(fig.layout, 1, 2)
    for r_idx in 2:n_rows
        rowgap!(fig.layout, r_idx, 4)
    end
    rowgap!(fig.layout, n_rows + 1, 0)
    rowgap!(fig.layout, n_rows + 2, 4)
    colgap!(fig.layout, 0)

    return fig
end

# ===========================================================================
# Default sweep — same grid and configs as the paper figure.
# ===========================================================================

const E_TRANS_eV_LIST  = [0.2, 0.5, 1.0]
const ν_INITIAL_LIST   = [3, 16]
const T_K_LIST         = [300]

const _ω_GRID_eV = vcat(0.0,
    [1e-7, 3.16e-7, 1e-6, 3.16e-6, 1e-5, 3.16e-5,
     1e-4, 3.16e-4, 1e-3, 3.16e-3],
    collect(0.01:0.01:20.0))

memory_endpt_configs = [Dict{String,Any}(
        "model"          => :NOAu,
        "T_K"            => T,
        "ω"              => _ω_GRID_eV,
        "stride"         => 1,
        "parallel"       => false,
        "kernel_average" => :endpoint,
    ) for T in T_K_LIST]

markovian_configs = [Dict{String,Any}(
        "model"    => :NOAu,
        "T_K"      => T,
        "stride"   => 1,
        "parallel" => false,
    ) for T in T_K_LIST]

configs_noau = vcat(memory_endpt_configs, markovian_configs)

# σ = Gaussian width in units of ν: smaller resolves finer structure, larger
# smooths towards the overall envelope.
fig = plot_vib_state_distribution_kde(ν_INITIAL_LIST, E_TRANS_eV_LIST, configs_noau;
                                      T_focus = 300, σ = 0.1)
display(fig)

save(plotsdir("cpa", "NOAu", "vib_state_distribution_kde.png"), fig)
