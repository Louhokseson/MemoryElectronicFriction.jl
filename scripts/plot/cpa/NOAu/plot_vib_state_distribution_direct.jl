using DrWatson
@quickactivate "HokseonReproduce"

using HDF5
using LinearAlgebra: eigen
using Unitful, UnitfulAtomic
using StaticArrays: SA
using CairoMakie
using ColorSchemes

using HokseonReproduce
using HokseonPlots
using HokseonAssistant
using QuadGK: quadgk
using Roots: find_zero

using NQCModels
# NB: no NQCDynamics.QuantisedDiatomic imports — we deliberately skip the
# B-spline `binding_curve` / `EffectivePotential` machinery and call the bare
# 1D adiabatic potential directly. See `quantise_from_energy` below.

# ---------------------------------------------------------------------------
# 1D adiabatic surface for the NO bond — identical to
# `plot_vib_state_distribution.jl`. We freeze the molecule–surface height at
# z = 10 Å (same value `noau_initial_distribution` in `run_md.jl` uses) so the
# ν we recover from the final (r, ṙ) lives on the same potential the initial
# state was EBK-sampled from.
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

# ---------------------------------------------------------------------------
# Direct 1D potential — no B-spline interpolant.
#
# Wraps the bare callable r → V₁(r) (one 2×2 `eigen` per evaluation) plus
# cached well metadata (r_eq, V_min, an outer-wall reference V_outer, and the
# inner/outer brackets we'll use for turning-point root finding).
#
# Compared to the EffectivePotential + binding_curve path in
# `plot_vib_state_distribution.jl`, this trades a one-time spline fit for
# repeated `eigen` calls. Each call is ~10 μs on a 2×2 matrix, so the cost
# is negligible at our trajectory counts (~10⁴ `quantise_from_energy` calls).
# ---------------------------------------------------------------------------

struct DirectPotential{F, T<:Real}
    V1              :: F   # callable V₁(r::Real)::Real (atomic units)
    μ               :: T
    r_eq            :: T
    V_min           :: T
    V_outer         :: T   # V₁ evaluated at r_outer_bracket — outer wall reference
    r_inner_bracket :: T
    r_outer_bracket :: T
end

# Make `DirectPotential` callable so `V(r) == V.V1(r)` (matches how the
# original script used `V_eff(r_f)`).
(V::DirectPotential)(r) = V.V1(r)

function build_direct_potential(; z_eval_Å::Real = 10.0,
                                  r_inner_Å::Real = 0.5,
                                  r_outer_Å::Real = 5.0,
                                  r_eq_grid_Å      = 0.7:0.005:2.0)
    μ_NO  = austrip((14.007 * 15.999 / (14.007 + 15.999)) * u"u")
    bo_2d = BOAdiabaticModel(POGOModel(), 1)
    bo_1d = FrozenHeightBO(bo_2d, austrip(z_eval_Å * u"Å"))

    # bare 1D potential. `[r;;]` is shorthand for `reshape([r], 1, 1)`.
    V1(r) = NQCModels.potential(bo_1d, [r;;])

    # Locate r_eq: coarse grid scan to bracket the minimum, then refine with
    # `find_zero` on a centred-difference derivative.
    rs_au   = austrip.(r_eq_grid_Å .* u"Å")
    Vs      = V1.(rs_au)
    idx_min = argmin(Vs)
    @assert 1 < idx_min < length(rs_au) "well minimum at the r_eq_grid_Å boundary — extend the grid"
    h       = 1e-5
    dV(r)   = (V1(r + h) - V1(r - h)) / (2h)
    r_eq    = find_zero(dV, (rs_au[idx_min - 1], rs_au[idx_min + 1]))
    V_min   = V1(r_eq)

    r_inner_au = austrip(r_inner_Å * u"Å")
    r_outer_au = austrip(r_outer_Å * u"Å")
    V_outer    = V1(r_outer_au)

    return DirectPotential(V1, μ_NO, r_eq, V_min, V_outer, r_inner_au, r_outer_au)
end

# ---------------------------------------------------------------------------
# EBK quantisation from a given total bond energy — direct (no B-spline).
#
#   n_r(E) = √(2μ)/π · ∫_{r₁}^{r₂} √(E − V(r)) dr − 1/2,
#
# with turning points r₁, r₂ solving V(r) = E. Sentinels match the values
# returned by `plot_vib_state_distribution.jl` so downstream histogram /
# rounding code is reused unchanged.
# ---------------------------------------------------------------------------

const _NU_BELOW_WELL  = -1   # E below the well minimum
const _NU_DISSOCIATED = -2   # E above the outer-wall reference

function quantise_from_energy(E::Real, V::DirectPotential)
    E ≤ V.V_min   && return _NU_BELOW_WELL
    E ≥ V.V_outer && return _NU_DISSOCIATED
    r₁ = find_zero(r -> V.V1(r) - E, (V.r_inner_bracket, V.r_eq))
    r₂ = find_zero(r -> V.V1(r) - E, (V.r_eq,            V.r_outer_bracket))
    integral, _ = quadgk(r -> sqrt(max(E - V.V1(r), 0.0)), r₁, r₂; maxevals = 200)
    return sqrt(2 * V.μ) / π * integral - 1 / 2
end

# ---------------------------------------------------------------------------
# Final-state pipeline for one (CPA config, MD params) cell.
#
# Steps per trajectory i:
#   1. Read final bond (r, ṙ) from the MD trajectory (column `end`).
#   2. Form the MD-only bond energy E_bond_MD = ½μṙ² + V₁(r).
#   3. Subtract ΔE_r[i] (CPA's vibrational energy loss for that trajectory).
#   4. EBK-quantise both energies on the same direct potential.
# ---------------------------------------------------------------------------

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

    ν_md  = Vector{Float64}(undef, n)
    ν_cpa = Vector{Float64}(undef, n)
    E_md  = Vector{Float64}(undef, n)
    for i in 1:n
        r_f = trajs[i].OutputPosition[1, end]
        v_f = trajs[i].OutputVelocity[1, end]
        E_bond = 0.5 * V.μ * v_f^2 + V(r_f)       # J=0 → V(r) = V_1d(r)
        E_md[i]  = E_bond
        ν_md[i]  = quantise_from_energy(E_bond,            V)
        ν_cpa[i] = quantise_from_energy(E_bond - ΔE_r[i],  V)
    end
    return (ν_md = ν_md, ν_cpa = ν_cpa,
            ΔE_r_au = Vector(ΔE_r[1:n]), E_md_au = E_md)
end

# ---------------------------------------------------------------------------
# P(ν_final) — integer-bin histogram. Sentinels (below-well, dissociated) are
# tallied as separate columns so the user can see how many trajectories the
# EBK procedure rejects, without inflating physical bins.
# ---------------------------------------------------------------------------

function vib_histogram(ν_continuous::AbstractVector{<:Real}, ν_max::Integer)
    counts = zeros(Int, ν_max + 1)   # bins 0..ν_max
    rejected_below  = 0
    rejected_disso  = 0
    rejected_high   = 0
    for νc in ν_continuous
        if νc == _NU_BELOW_WELL
            rejected_below += 1
        elseif νc == _NU_DISSOCIATED
            rejected_disso += 1
        else
            νi = round(Int, νc)
            νi < 0          && (rejected_below += 1; continue)
            νi > ν_max      && (rejected_high  += 1; continue)
            counts[νi + 1] += 1
        end
    end
    total = length(ν_continuous)
    prob  = counts ./ total
    return (ν = collect(0:ν_max), prob = prob, counts = counts,
            n_total = total,
            n_below = rejected_below, n_disso = rejected_disso,
            n_high  = rejected_high)
end

# 1σ binomial (Wald) standard error on a proportion p estimated from N trials.
binomial_se(prob::AbstractVector{<:Real}, n_total::Integer) =
    sqrt.(max.(prob .* (1 .- prob) ./ max(n_total, 1), 0.0))

# ---------------------------------------------------------------------------
# Variant tags — same logic plot_vib_DeltaE.jl uses (memory CPA carries
# `kernel_average`; Markovian does not).
# ---------------------------------------------------------------------------

variant_of(cfg) = haskey(cfg, "kernel_average") ? :memory : :markovian
variant_label(::Val{:memory})    = "memory CPA"
variant_label(::Val{:markovian}) = "Markovian CPA"

# ---------------------------------------------------------------------------
# MD params builder — mirrors run_md.jl's `all_params_NOAu` so the savename
# round-trips. One Eₜ and one ν_initial per call.
# ---------------------------------------------------------------------------

function build_md_params(E_TRANS_eV::Real, ν_initial::Integer)
    return Dict{String, Any}(
        "mass"                  => (14.007 * 15.999 / (14.007 + 15.999)) * u"u",
        "r0"                    => [1.15u"Å", 5.0u"Å"],
        "translational_kinetic" => E_TRANS_eV * u"eV",
        "state"                 => 1,
        "tmax"                  => 500.0u"fs",
        "dt"                    => 0.25u"fs",
        "termination_min_time"  => 10.0u"fs",
        "termination_coord_idx" => 2,
        "termination_threshold" => 5.0u"Å",
        "vibrational_state"     => ν_initial,
        "trajectories"          => 1000,
    )
end

# ---------------------------------------------------------------------------
# Figure layout: rows = ν_initial, cols = Eₜ. Each panel compares two methods
# at a single temperature (Memory vs Markovian CPA) as connected lines with
# circle markers and 1σ binomial error bars. A dashed vertical guide marks
# ν_initial. Mirrors the visual style of Wodtke/Tully-type P(ν_final) plots.
# ---------------------------------------------------------------------------

function plot_vib_state_distribution_direct(ν_INITIAL_LIST, E_TRANS_eV_LIST,
                                             configs::AbstractVector;
                                             T_focus          = 1000,
                                             ν_max            = 20,
                                             figsize          = (HokseonPlots.RESOLUTION[1] * 2.2,
                                                                 HokseonPlots.RESOLUTION[2] * 1.5 * length(ν_INITIAL_LIST)),
                                             panel_spinewidth = 1.4,
                                             show_md_baseline = false,
                                             ylims            = (-0.05, nothing))

    V = build_direct_potential()
    @info "Built direct 1D potential" V_min_eV=ustrip(auconvert(u"eV", V.V_min)) r_eq_Å=ustrip(auconvert(u"Å", V.r_eq)) V_outer_eV=ustrip(auconvert(u"eV", V.V_outer))

    # Pick the two configs to compare in every panel: memory + Markovian at T_focus.
    mem_cfg  = nothing
    mark_cfg = nothing
    for c in configs
        c["T_K"] == T_focus || continue
        if variant_of(c) == :memory && mem_cfg === nothing
            mem_cfg = c
        elseif variant_of(c) == :markovian && mark_cfg === nothing
            mark_cfg = c
        end
    end
    mem_cfg  === nothing && error("No memory CPA config at T = $(T_focus) K in `configs`")
    mark_cfg === nothing && error("No Markovian CPA config at T = $(T_focus) K in `configs`")

    n_rows = length(ν_INITIAL_LIST)
    n_cols = length(E_TRANS_eV_LIST)

    # Two-method palette — orange / cyan-blue, matching the reference figure.
    color_mem  = "#E89A3C"
    color_mark = "#3F9DCC"
    color_md   = "#444444"

    fig = Figure(; size = figsize,
                   figure_padding = (8, 12, 6, 6),
                   fonts = (; regular = projectdir("fonts", "MinionPro-Capt.otf")))

    # Column headers (Eₜ tags).
    for (c, Eₜ) in enumerate(E_TRANS_eV_LIST)
        Label(fig[1, c], "Eₜ = $(Eₜ) eV";
              fontsize   = 14,
              halign     = :center,
              tellwidth  = false,
              tellheight = true,
              padding    = (0, 0, 2, 0))
    end

    # Per-row x-spec: each ν_initial row gets its own x-range and tick set.
    function row_xspec(ν_init)
        if ν_init ≤ 5
            return (xticks = 0:1:5,         xlims = (-0.5, 5.5))
        elseif ν_init ≤ 10
            return (xticks = 0:2:10,        xlims = (-0.5, 10.5))
        else
            return (xticks = 0:4:ν_max,     xlims = (-1, ν_max + 0.5))
        end
    end

    axes = Matrix{Any}(undef, n_rows, n_cols)
    for (r_idx, ν_init) in enumerate(ν_INITIAL_LIST)
        spec = row_xspec(ν_init)
        for (c_idx, Eₜ) in enumerate(E_TRANS_eV_LIST)
            ax = Axis(fig[r_idx + 1, c_idx];
                      xticks = spec.xticks,
                      xtickalign = 1.0,
                      ytickalign = 1.0,
                      xticksmirrored = true,
                      yticksmirrored = true,
                      limits = (spec.xlims[1], spec.xlims[2], ylims[1], ylims[2]),
                      spinewidth = panel_spinewidth,
                      xtickwidth = panel_spinewidth,
                      ytickwidth = panel_spinewidth,
                      xgridvisible = false, ygridvisible = false,
                      xminorticksvisible = false, yminorticksvisible = false)

            params_md = build_md_params(Eₜ, ν_init)
            res_mem  = final_vib_states(params_md, mem_cfg,  V)
            res_mark = final_vib_states(params_md, mark_cfg, V)

            # vertical dashed guide at ν_initial
            vlines!(ax, [ν_init]; color = :black, linestyle = :dash,
                    linewidth = 1.0)

            if show_md_baseline
                md_res = res_mem !== nothing ? res_mem :
                         res_mark !== nothing ? res_mark : nothing
                if md_res !== nothing
                    h = vib_histogram(md_res.ν_md, ν_max)
                    err = binomial_se(h.prob, h.n_total)
                    errorbars!(ax, h.ν, h.prob, err;
                               color = color_md, whiskerwidth = 8, linewidth = 1.4)
                    scatterlines!(ax, h.ν, h.prob;
                                  color = color_md,
                                  linewidth = 1.2,
                                  markersize = 7,
                                  marker = :circle,
                                  strokecolor = :black,
                                  strokewidth = 1.2)
                end
            end

            for (res, col) in ((res_mem,  color_mem),
                               (res_mark, color_mark))
                res === nothing && continue
                h   = vib_histogram(res.ν_cpa, ν_max)
                err = binomial_se(h.prob, h.n_total)
                errorbars!(ax, h.ν, h.prob, err;
                           color = col, whiskerwidth = 8, linewidth = 1.4)
                scatterlines!(ax, h.ν, h.prob;
                              color = col,
                              linewidth = 1.6,
                              markersize = 8,
                              marker = :circle,
                              strokecolor = :black,
                              strokewidth = 1.2)
            end

            axes[r_idx, c_idx] = ax
        end
    end

    # Per-row y-link so probabilities are directly comparable across Eₜ.
    for r_idx in 1:n_rows
        linkyaxes!(axes[r_idx, :]...)
    end
    # Inner columns drop y-tick labels (panels are flush — no column gap).
    for r_idx in 1:n_rows, c_idx in 2:n_cols
        hideydecorations!(axes[r_idx, c_idx]; grid = false, ticks = false)
    end

    # Shared axis labels.
    Label(fig[:, 0], "Probability";
          fontsize = 14,
          rotation = π / 2,
          tellheight = false,
          padding = (0, 4, 0, 0))
    Label(fig[n_rows + 2, 1:n_cols], "Final vibrational state ν";
          fontsize = 14,
          tellwidth = false,
          padding = (0, 0, 0, 0))

    # Shared legend at the bottom.
    legend_elems  = Any[
        [LineElement(;   color = color_mem,  linewidth = 1.6),
         MarkerElement(; color = color_mem,  marker = :circle, markersize = 10,
                         strokecolor = :black, strokewidth = 1.2)],
        [LineElement(;   color = color_mark, linewidth = 1.6),
         MarkerElement(; color = color_mark, marker = :circle, markersize = 10,
                         strokecolor = :black, strokewidth = 1.2)],
        LineElement(; color = :black, linestyle = :dash, linewidth = 1.0),
    ]
    legend_labels = String[
        "Memory CPA",
        "Markovian CPA",
        "Initial vibrational state",
    ]
    if show_md_baseline
        pushfirst!(legend_elems,
            [MarkerElement(; color = color_md, marker = :circle, markersize = 10,
                             strokecolor = :white, strokewidth = 0.8),
             LineElement(;   color = color_md, linewidth = 1.2)])
        pushfirst!(legend_labels, "MD (no friction)")
    end

    Legend(fig[n_rows + 3, 1:n_cols], legend_elems, legend_labels;
           orientation = :horizontal,
           tellwidth = false,
           tellheight = true,
           framevisible = true,
           framecolor = :black,
           framewidth = panel_spinewidth,
           labelsize = 11,
           patchsize = (24, 12),
           margin = (4, 4, 2, 2))

    rowgap!(fig.layout, 1, 2)
    for r_idx in 2:n_rows
        rowgap!(fig.layout, r_idx, 4)
    end
    rowgap!(fig.layout, n_rows + 1, 0)
    rowgap!(fig.layout, n_rows + 2, 4)
    # Flush axis panels in the same row; shared y-label keeps breathing room
    # via its own `padding = (0, 4, 0, 0)`.
    colgap!(fig.layout, 0)

    return fig
end

# ===========================================================================
# Default sweep — same grid as plot_vib_state_distribution.jl so the two
# versions can be compared panel-by-panel.
# ===========================================================================

const E_TRANS_eV_LIST  = [0.2, 1.0]
const ν_INITIAL_LIST   = [3, 16]
const T_K_LIST         = [300, 1000, 2000]

const _ω_GRID_eV = vcat(0.0,
    [1e-7, 3.16e-7, 1e-6, 3.16e-6, 1e-5, 3.16e-5,
     1e-4, 3.16e-4, 1e-3, 3.16e-3],
    collect(0.01:0.01:20.0))

memory_configs = [Dict{String,Any}(
        "model"          => :NOAu,
        "T_K"            => T,
        "ω"              => _ω_GRID_eV,
        "stride"         => 1,
        "parallel"       => false,
        "kernel_average" => :arithmetic,
    ) for T in T_K_LIST]

markovian_configs = [Dict{String,Any}(
        "model"   => :NOAu,
        "T_K"     => T,
        "stride"  => 1,
        "parallel" => false,
    ) for T in T_K_LIST]

configs_noau = vcat(memory_configs, markovian_configs)

fig = plot_vib_state_distribution_direct(ν_INITIAL_LIST, E_TRANS_eV_LIST, configs_noau;
                                          ν_max = 20)
display(fig)

# Optional save — uncomment to render to disk.
# save_figure(plotsdir("cpa", "NOAu", "vib_state_distribution_direct"), fig)
