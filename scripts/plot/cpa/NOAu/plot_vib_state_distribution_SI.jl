using DrWatson
@quickactivate "MemoryElectronicFriction"

using HDF5
using LinearAlgebra: eigen
using Unitful, UnitfulAtomic
using StaticArrays: SA
using CairoMakie
using ColorSchemes

using MemoryElectronicFriction
using HokseonPlots
using HokseonAssistant
using QuadGK: quadgk
using Roots: find_zero

using NQCModels
# NB: no NQCDynamics.QuantisedDiatomic imports — we deliberately skip the
# B-spline `binding_curve` / `EffectivePotential` machinery and call the bare
# 1D adiabatic potential directly. See `quantise_from_energy` below.
#
# This file is a sibling of `plot_vib_state_distribution_direct.jl`. The
# difference is the comparison axis: instead of memory CPA vs Markovian CPA at
# a fixed T, here both lines are *memory* CPA at a fixed T, and we vary the
# kernel-time-averaging scheme (`:arithmetic` vs `:endpoint`).

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
# Direct 1D potential — no B-spline interpolant. See header doc in
# `plot_vib_state_distribution_direct.jl` for the rationale.
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

    r_inner_au = austrip(r_inner_Å * u"Å")
    r_outer_au = austrip(r_outer_Å * u"Å")
    V_outer    = V1(r_outer_au)

    return DirectPotential(V1, μ_NO, r_eq, V_min, V_outer, r_inner_au, r_outer_au)
end

# ---------------------------------------------------------------------------
# EBK quantisation from a given total bond energy — direct (no B-spline).
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
# P(ν_final) — integer-bin histogram + binomial SE. Identical to the sibling
# direct script.
# ---------------------------------------------------------------------------

function vib_histogram(ν_continuous::AbstractVector{<:Real}, ν_max::Integer)
    counts = zeros(Int, ν_max + 1)
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

binomial_se(prob::AbstractVector{<:Real}, n_total::Integer) =
    sqrt.(max.(prob .* (1 .- prob) ./ max(n_total, 1), 0.0))

# ---------------------------------------------------------------------------
# Variant tags. Three configs live in the same `configs` list:
#   - memory CPA with `kernel_average=:arithmetic`
#   - memory CPA with `kernel_average=:endpoint`
#   - Markovian CPA (no `kernel_average` key — matches plot_vib_DeltaE.jl)
# ---------------------------------------------------------------------------

kernel_avg_of(cfg) = get(cfg, "kernel_average", nothing)
is_markovian(cfg)  = !haskey(cfg, "kernel_average")

# ---------------------------------------------------------------------------
# MD params builder — mirrors run_md.jl's `all_params_NOAu`.
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
# Figure layout: rows = ν_initial, cols = Eₜ. Each panel compares
# memory CPA (arithmetic) vs memory CPA (endpoint) at a single temperature.
# ---------------------------------------------------------------------------

function plot_vib_state_distribution_compare(ν_INITIAL_LIST, E_TRANS_eV_LIST,
                                              configs::AbstractVector;
                                              T_focus          = 300,
                                              ν_max            = 20,
                                              # Full text width (≈ 6.68 in / 170 mm) so it fits a
                                              # single-column SI page at width=\linewidth. Height kept
                                              # well below the width (landscape) so this 2×3 grid does
                                              # not dominate the page — ratio 0.68 → ~4.5 in tall.
                                              figsize          = (HokseonPlots.TWO_COLUMN_WIDTH,
                                                                  HokseonPlots.TWO_COLUMN_WIDTH * 0.75),
                                              panel_spinewidth = 1.8,
                                              show_md_baseline = false,
                                              ylims            = (-0.05, nothing),
                                              # Font hierarchy (pt at the authored full-width size).
                                              # Scaled ~0.7× from the original 18–20 pt set so they
                                              # sit closer to PRX 10 pt body text at width=\linewidth.
                                              title_fontsize   = 14,   # panel Eₜ headers
                                              label_fontsize   = 14,   # axis labels (Probability / Final vibrational state ν)
                                              tick_fontsize    = 12,   # x / y tick labels
                                              legend_fontsize  = 12,
                                              panel_label_fontsize = 13)  # (a)–(f) tags

    V = build_direct_potential()
    @info "Built direct 1D potential" V_min_eV=ustrip(auconvert(u"eV", V.V_min)) r_eq_Å=ustrip(auconvert(u"Å", V.r_eq)) V_outer_eV=ustrip(auconvert(u"eV", V.V_outer))

    # Pick the three configs at T_focus: two memory CPA (one per kernel
    # averaging scheme) plus one Markovian CPA.
    arith_cfg = nothing
    endpt_cfg = nothing
    mark_cfg  = nothing
    for c in configs
        c["T_K"] == T_focus || continue
        if is_markovian(c) && mark_cfg === nothing
            mark_cfg = c
        else
            ka = kernel_avg_of(c)
            if ka == :arithmetic && arith_cfg === nothing
                arith_cfg = c
            elseif ka == :endpoint && endpt_cfg === nothing
                endpt_cfg = c
            end
        end
    end
    arith_cfg === nothing && error("No arithmetic-kernel memory CPA config at T = $(T_focus) K in `configs`")
    endpt_cfg === nothing && error("No endpoint-kernel memory CPA config at T = $(T_focus) K in `configs`")
    mark_cfg  === nothing && error("No Markovian CPA config at T = $(T_focus) K in `configs`")

    n_rows = length(ν_INITIAL_LIST)
    n_cols = length(E_TRANS_eV_LIST)

    # Three-method palette — arithmetic = orange (matches the sibling script's
    # "memory CPA" colour), endpoint = magenta, Markovian = cyan-blue (matches
    # the sibling script's Markovian colour).
    color_arith = "#E89A3C"
    color_endpt = "#B53A8C"
    color_mark  = "#3F9DCC"
    color_md    = "#444444"

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
            return (xticks = 0:1:5,         xlims = (-0.5, 5.5))
        elseif ν_init ≤ 10
            return (xticks = 0:2:10,        xlims = (-0.5, 10.5))
        else
            # Bottom row: start at ν = 6 (the low-ν tail is empty) and run ticks
            # out to 18, leaving a little margin to the right of the
            # initial-state line at ν = νᵢ = 16. Data lives ≈ν 7–13.
            return (xticks = 6:3:18,        xlims = (5, 19))
        end
    end

    axes = Matrix{Any}(undef, n_rows, n_cols)
    for (r_idx, ν_init) in enumerate(ν_INITIAL_LIST)
        spec = row_xspec(ν_init)
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
                      spinewidth = panel_spinewidth,
                      xtickwidth = panel_spinewidth,
                      ytickwidth = panel_spinewidth,
                      xgridvisible = false, ygridvisible = false,
                      xminorticksvisible = false, yminorticksvisible = false)

            params_md = build_md_params(Eₜ, ν_init)
            res_arith = final_vib_states(params_md, arith_cfg, V)
            res_endpt = final_vib_states(params_md, endpt_cfg, V)
            res_mark  = final_vib_states(params_md, mark_cfg,  V)

            vlines!(ax, [ν_init]; color = :black, linestyle = :dash,
                    linewidth = 1.0)

            if show_md_baseline
                md_res = res_arith !== nothing ? res_arith :
                         res_endpt !== nothing ? res_endpt :
                         res_mark  !== nothing ? res_mark  : nothing
                if md_res !== nothing
                    h = vib_histogram(md_res.ν_md, ν_max)
                    err = binomial_se(h.prob, h.n_total)
                    errorbars!(ax, h.ν, h.prob, err;
                               color = color_md, whiskerwidth = 12, linewidth = 2.0)
                    scatterlines!(ax, h.ν, h.prob;
                                  color = color_md,
                                  linewidth = 1.2,
                                  markersize = 9,
                                  marker = :circle,
                                  strokecolor = :black,
                                  strokewidth = 1.2)
                end
            end

            for (res, col, mrk) in ((res_arith, color_arith, :utriangle),
                                    (res_endpt, color_endpt, :diamond),
                                    (res_mark,  color_mark,  :circle))
                res === nothing && continue
                h   = vib_histogram(res.ν_cpa, ν_max)
                err = binomial_se(h.prob, h.n_total)
                errorbars!(ax, h.ν, h.prob, err;
                           color = col, whiskerwidth = 12, linewidth = 2.0)
                scatterlines!(ax, h.ν, h.prob;
                              color = col,
                              linewidth = 1.6,
                              markersize = 9,
                              marker = mrk,
                              strokecolor = :black,
                              strokewidth = 1.2)
            end

            # Panel letter (a)–(f), row-major, top-left inset. Faux-bold via a
            # thin same-colour stroke keeps MinionPro (no bold face ships with
            # the project) — see plot_LambdaElement_r.jl.
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

    # Right-side row labels: the initial vibrational state νᵢ for each row.
    #for (r_idx, ν_init) in enumerate(ν_INITIAL_LIST)
    #    Label(fig[r_idx + 1, n_cols + 1], rich("ν", subscript("i"), " = $(ν_init)");
    #          fontsize   = label_fontsize,
    #          rotation   = π / 2,
    #          tellheight = false,
    #          padding    = (6, 2, 0, 0))
    #end

    Label(fig[:, 0], "Probability";
          fontsize = label_fontsize,
          rotation = π / 2,
          tellheight = false,
          padding = (0, 4, 0, 0))
    Label(fig[n_rows + 2, 1:n_cols], "Final vibrational state ν";
          fontsize = label_fontsize,
          tellwidth = false,
          padding = (0, 0, 0, 0))

    legend_elems  = Any[
        [LineElement(;   color = color_arith, linewidth = 1.6),
         MarkerElement(; color = color_arith, marker = :utriangle, markersize = 10,
                         strokecolor = :black, strokewidth = 1.2)],
        [LineElement(;   color = color_endpt, linewidth = 1.6),
         MarkerElement(; color = color_endpt, marker = :diamond, markersize = 10,
                         strokecolor = :black, strokewidth = 1.2)],
        [LineElement(;   color = color_mark,  linewidth = 1.6),
         MarkerElement(; color = color_mark,  marker = :circle, markersize = 10,
                         strokecolor = :black, strokewidth = 1.2)],
        LineElement(; color = :black, linestyle = :dash, linewidth = 1.0),
    ]
    legend_labels = String[
        "Memory (arithmetic)",
        "Memory (local)",
        "Markovian",
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
           nbanks = 2,
           tellwidth = false,
           tellheight = true,
           framevisible = false,
           framecolor = :black,
           framewidth = panel_spinewidth,
           labelsize = legend_fontsize,
           patchsize = (24, 12),
           margin = (4, 4, 2, 2))

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
# Default sweep — focus on T = 300 K, where both arithmetic and endpoint
# kernel-averaged memory CPA runs exist for the (Eₜ, ν_initial) grid below.
# ===========================================================================

const E_TRANS_eV_LIST  = [0.2, 0.5, 1.0]
const ν_INITIAL_LIST   = [3, 16]
const T_K_LIST         = [300]

const _ω_GRID_eV = vcat(0.0,
    [1e-7, 3.16e-7, 1e-6, 3.16e-6, 1e-5, 3.16e-5,
     1e-4, 3.16e-4, 1e-3, 3.16e-3],
    collect(0.01:0.01:20.0))

memory_arith_configs = [Dict{String,Any}(
        "model"          => :NOAu,
        "T_K"            => T,
        "ω"              => _ω_GRID_eV,
        "stride"         => 1,
        "parallel"       => false,
        "kernel_average" => :arithmetic,
    ) for T in T_K_LIST]

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

configs_noau = vcat(memory_arith_configs, memory_endpt_configs, markovian_configs)

fig = plot_vib_state_distribution_compare(ν_INITIAL_LIST, E_TRANS_eV_LIST, configs_noau;
                                           T_focus = 300, ν_max = 18)
display(fig)

# Optional save — uncomment to render to disk.
save(plotsdir("cpa", "NOAu", "vib_state_distribution_SI.pdf"), fig)
