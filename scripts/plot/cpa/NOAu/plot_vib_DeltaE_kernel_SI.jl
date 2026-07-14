using Distributed
using DrWatson
@quickactivate "MemoryElectronicFriction"

using Unitful, UnitfulAtomic
using LinearAlgebra: eigen
using StaticArrays: SA
using Random
using HDF5
using HokseonAssistant
using HokseonPlots
using CairoMakie
using Colors

HokseonAssistant.julia_build_procs()
@everywhere using MemoryElectronicFriction
@everywhere using StaticArrays: SA
@everywhere using Unitful, UnitfulAtomic

# ---------------------------------------------------------------------------
# Kernel-average detection. Memory CPA carries `kernel_average`; Markovian
# does not. We classify each config into one of three methods:
#   :arith      — memory CPA with kernel_average = :arithmetic
#   :endpoint   — memory CPA with kernel_average = :endpoint
#   :markovian  — Markovian CPA (no kernel_average key)
# ---------------------------------------------------------------------------
kernel_avg_of(cfg) = get(cfg, "kernel_average", nothing)

function method_of(cfg)
    ka = kernel_avg_of(cfg)
    ka === nothing      ? :markovian :
    ka == :arithmetic   ? :arith     :
    ka == :endpoint     ? :endpoint  :
    error("Unknown kernel_average: $ka")
end

method_label(::Val{:arith})     = "memory (arithmetic)"
method_label(::Val{:endpoint})  = "memory (endpoint)"
method_label(::Val{:markovian}) = "Markovian"

# ---------------------------------------------------------------------------
# Read one (params_list, cfg) sweep into per-DOF (ν, ΔE_mean, ΔE_lo, ΔE_hi)
# vectors. NOAu CPA writes a 2 × n_traj matrix per file with rows
#   row 1 = ΔE_r (NO bond-stretch, *vibrational* loss),
#   row 2 = ΔE_z (NO–surface separation, *translational* loss).
# We report ensemble means with the 2.5–97.5 percentile interval (95% interval
# of the trajectory distribution, not the CI of the mean). Asymmetric (lo, hi)
# offsets don't assume the distribution is Gaussian. Skips and warns on
# missing files so a partially-finished sweep still plots cleanly.
# ---------------------------------------------------------------------------
function load_vib_DeltaE(params_list, cfg)
    ν_list    = Int[]
    vib_mean  = Float64[]
    vib_lo    = Float64[]
    vib_hi    = Float64[]
    tra_mean  = Float64[]
    tra_lo    = Float64[]
    tra_hi    = Float64[]
    for p in params_list
        savingpath, savingname = CPA_dict_to_data_savename(p, cfg)
        full_data_path = datadir(savingpath, savingname)
        if !isfile(full_data_path)
            @warn "Missing CPA data file" full_data_path; continue
        end
        ΔE_au_mat = h5open(full_data_path, "r") do f
            read(f, "DeltaE_au")
        end
        # Skip legacy single-DOF files — this plot needs the 2-DOF decomposition.
        if !(ndims(ΔE_au_mat) == 2 && size(ΔE_au_mat, 1) == 2)
            @warn "DeltaE_au is not a 2 × n_traj matrix — skipping" full_data_path size=size(ΔE_au_mat)
            continue
        end
        ΔE_r_au = ΔE_au_mat[1, :]
        ΔE_z_au = ΔE_au_mat[2, :]

        push!(ν_list, p["vibrational_state"])

        m_r  = mean(ΔE_r_au)
        q1_r = quantile(ΔE_r_au, 0.025)
        q9_r = quantile(ΔE_r_au, 0.975)
        push!(vib_mean, ustrip(auconvert(u"eV", m_r)))
        push!(vib_lo,   ustrip(auconvert(u"eV", m_r - q1_r)))
        push!(vib_hi,   ustrip(auconvert(u"eV", q9_r - m_r)))

        m_z  = mean(ΔE_z_au)
        q1_z = quantile(ΔE_z_au, 0.025)
        q9_z = quantile(ΔE_z_au, 0.975)
        push!(tra_mean, ustrip(auconvert(u"eV", m_z)))
        push!(tra_lo,   ustrip(auconvert(u"eV", m_z - q1_z)))
        push!(tra_hi,   ustrip(auconvert(u"eV", q9_z - m_z)))
    end
    order = sortperm(ν_list)
    return (ν   = ν_list[order],
            vib = (vib_mean[order], vib_lo[order], vib_hi[order]),
            tra = (tra_mean[order], tra_lo[order], tra_hi[order]))
end

# Local mean/quantile to avoid a Statistics dependency.
mean(xs) = sum(xs) / length(xs)
function quantile(xs, p)
    sorted = sort(xs)
    n = length(sorted)
    n == 0 && return 0.0
    n == 1 && return sorted[1]
    h  = (n - 1) * p + 1
    lo = clamp(floor(Int, h), 1, n)
    hi = clamp(ceil(Int, h),  1, n)
    return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo])
end

# ---------------------------------------------------------------------------
# Local clone of HokseonPlots.MyAxis with a `mirror_top` switch.
# ---------------------------------------------------------------------------
function panel_axis(f; mirror_top::Bool = true, kwargs...)
    main  = Axis(f; kwargs...)
    extra = Axis(f; xaxisposition = :top, yaxisposition = :right, kwargs...)
    linkaxes!(main, extra)
    hidespines!(extra)
    hidedecorations!(extra; ticks = false, minorticks = false)
    if !mirror_top
        extra.xticksvisible      = false
        extra.xminorticksvisible = false
    end
    return main
end

# ---------------------------------------------------------------------------
# Render one (x, ΔE) curve into `ax`. Marker mapping mirrors
# `plot_vib_state_distribution_compare.jl` (filled color marker + black
# stroke) so the two figures read as a set:
#
#   :arith     — solid  line + filled up-triangle
#   :endpoint  — dashed line + filled diamond
#   :markovian — dotted line + filled circle
#
# x is the *plot* coordinate (integer index for equal spacing), not ν itself.
# ---------------------------------------------------------------------------
function draw_curve!(ax, xs, ys, lo, hi, col, method::Symbol)
    linestyle, marker = if method == :arith
        (:solid, :utriangle)
    elseif method == :endpoint
        (:solid,  :diamond)
    else  # :markovian
        (:solid,   :circle)
    end
    lines!(ax, xs, ys;
           color = col, linewidth = 1.8, linestyle = linestyle)
    errorbars!(ax, xs, ys, lo, hi;
               color = col, whiskerwidth = 8, linewidth = 1.0)
    scatter!(ax, xs, ys;
             color = col, marker = marker, markersize = 9,
             strokecolor = :black, strokewidth = 1.2)
end

# ---------------------------------------------------------------------------
# Build a NOAu params_list for a single incident translational energy.
# ---------------------------------------------------------------------------
function build_params_list_noau(E_TRANS_eV, ν_LIST)
    all_params = Dict{String, Any}(
        "mass"                  => [(14.007 * 15.999 / (14.007 + 15.999)) * u"u"],   # μ_NO
        "r0"                    => [[1.15u"Å", 5.0u"Å"]],
        "translational_kinetic" => [E_TRANS_eV * u"eV"],
        "state"                 => [1],
        "tmax"                  => [500.0u"fs"],
        "dt"                    => [0.25u"fs"],
        "termination_min_time"  => [10.0u"fs"],
        "termination_coord_idx" => [2],
        "termination_threshold" => [5.0u"Å"],
        "vibrational_state"     => ν_LIST,
        "trajectories"          => [1000],
    )
    return dict_list(all_params)
end

# ---------------------------------------------------------------------------
# 2×2 small-multiples publication figure: ΔE vs ν at multiple Eₜ values,
# comparing arithmetic vs endpoint kernel averaging (T = 300 K only).
#
# Layout (rows × cols)
#   row 1 — header band: Eₜ column tags
#   row 2 — vibrational  panels (a, b)
#   row 3 — translational panels (c, d)
#   row 4 — shared x-axis label "νᵢ / initial vibrational state"
#   row 5 — legend (methods only, colour-coded)
#
# Conventions mirror plot_vib_DeltaE.jl so the two figures read as a set.
# ---------------------------------------------------------------------------
const COLOR_ARITH   = parse(Colorant, "#E89A3C")   # orange
const COLOR_ENDPT   = parse(Colorant, "#B53A8C")   # magenta
const COLOR_MARKOV  = parse(Colorant, "#3F9DCC")   # cyan-blue

method_color(m::Symbol) =
    m == :arith    ? COLOR_ARITH :
    m == :endpoint ? COLOR_ENDPT :
    m == :markovian ? COLOR_MARKOV :
    error("Unknown method: $m")

function plot_vib_DeltaE_kernel(ν_LIST, E_TRANS_eV_LIST, configs::AbstractVector;
                                # Full text width (≈ 6.68 in / 170 mm); height ratio 0.68 to match
                                # vib_state_distribution_SI.pdf so the two figures read as a set at
                                # width=\linewidth in the SI.
                                figsize          = (HokseonPlots.TWO_COLUMN_WIDTH,
                                                    HokseonPlots.TWO_COLUMN_WIDTH * 0.68),
                                index_label_pos  = (0.03, 0.92),
                                panel_spinewidth = 1.8,
                                ylims            = nothing,
                                column_title_size = 14)

    methods = unique(method_of.(configs))

    n_ticks = length(ν_LIST)
    xticks  = (collect(1:n_ticks), string.(ν_LIST))
    ν_to_x  = Dict(ν_LIST[i] => i for i in 1:n_ticks)

    panel_keys    = (:vib, :tra)
    n_rows = length(panel_keys)
    n_cols = length(E_TRANS_eV_LIST)

    fig = Figure(; size = figsize,
                   figure_padding = (8, 4, 6, 6),
                   fonts = (; regular = projectdir("fonts", "MinionPro-Capt.otf")))

    # ----- Row 1: header band — Eₜ column tags ----------------------------
    for (c, Eₜ) in enumerate(E_TRANS_eV_LIST)
        is_first  = c == 1
        is_last   = c == n_cols
        halign_   = is_first ? :center  : (is_last ? :center : :center)
        pad_left  = is_first ? 6 : 0
        pad_right = is_last  ? 6 : 0
        Label(fig[1, c], "Eₜ = $(Eₜ) eV";
              fontsize   = column_title_size,
              halign     = halign_,
              tellwidth  = false,
              tellheight = true,
              padding    = (pad_left, pad_right, 2, 0))
    end

    # ----- Rows 2 & 3: panel grid ----------------------------------------
    axes = Matrix{Any}(undef, n_rows, n_cols)
    for r in 1:n_rows
        is_bottom = r == n_rows
        ylo, yhi  = if ylims === nothing || ylims[r] === nothing
            (nothing, nothing)
        else
            ylims[r]
        end
        for c in 1:n_cols
            is_left = c == 1
            params_list = build_params_list_noau(E_TRANS_eV_LIST[c], ν_LIST)
            axis_kwargs = (
                ylabel             = "",
                xticklabelsize     = 12,
                yticklabelsize     = 12,
                xminorticksvisible = false,
                yminorticksvisible = true,
                xminorgridvisible  = false,
                yminorgridvisible  = false,
                xgridvisible       = false,
                ygridvisible       = false,
                xticks             = xticks,
                xlabel             = "",
                limits             = (0.5, n_ticks + 0.5, ylo, yhi),
                spinewidth         = panel_spinewidth,
                xtickwidth         = panel_spinewidth,
                ytickwidth         = panel_spinewidth,
                xminortickwidth    = panel_spinewidth,
                yminortickwidth    = panel_spinewidth,
                xtickalign         = is_bottom ? 0 : 1,
                xminortickalign    = is_bottom ? 0 : 1,
                ytickalign         = 1,
                yminortickalign    = 1,
            )
            ax = panel_axis(fig[r + 1, c]; mirror_top = false, axis_kwargs...)

            for cfg in configs
                data = load_vib_DeltaE(params_list, cfg)
                isempty(data.ν) && continue
                μ, lo, hi = getfield(data, panel_keys[r])
                keep = [haskey(ν_to_x, ν) for ν in data.ν]
                xs   = [ν_to_x[ν] for ν in data.ν[keep]]
                μ_   = μ[keep]; lo_ = lo[keep]; hi_ = hi[keep]
                isempty(xs) && continue
                draw_curve!(ax, xs, μ_, lo_, hi_, method_color(method_of(cfg)), method_of(cfg))
            end

            # Faux-bold "(a)"–"(f)" in the figure's MinionPro-Capt regular face
            # (a thin same-colour stroke) — matches plot_vib_state_distribution_SI.jl;
            # `font = :bold` would fall back to a non-MinionPro bold face.
            letter = Char('a' + (r - 1) * n_cols + (c - 1))
            text!(ax, index_label_pos[1], index_label_pos[2];
                  text        = "($(letter))",
                  space       = :relative,
                  align       = (:left, :top),
                  fontsize    = 13,
                  color       = :black,
                  strokewidth = 0.5,
                  strokecolor = :black)

            axes[r, c] = ax
        end
    end

    # ----- Linking ---------------------------------------------------------
    for r in 1:n_rows
        linkyaxes!(axes[r, :]...)
    end
    for c in 1:n_cols
        linkxaxes!(axes[:, c]...)
    end

    for r in 1:(n_rows - 1), c in 1:n_cols
        hidexdecorations!(axes[r, c]; grid = false, ticks = false, minorticks = false)
    end
    for r in 1:n_rows, c in 2:n_cols
        hideydecorations!(axes[r, c]; grid = false, ticks = false, minorticks = false)
    end

    # ----- Shared y-axis labels --------------------------------------------
    # Left side — shared "Energy loss / eV" spanning both panel rows.
    Label(fig[2:3, 0], "Energy loss (eV)";
          fontsize   = 14,
          rotation   = π / 2,
          tellheight = false,
          padding    = (0, 4, 0, 0))

    # Right side — row-specific mode labels. tellwidth=true so the column
    # sizes to fit the rotated label snugly instead of letting it overflow
    # into the figure padding.
    Label(fig[2, n_cols + 1], "Vibration";
          fontsize   = 14,
          rotation   = π / 2,
          tellheight = false,
          padding    = (2, 0, 0, 0))
    Label(fig[3, n_cols + 1], "Translation";
          fontsize   = 14,
          rotation   = π / 2,
          tellheight = false,
          padding    = (2, 0, 0, 0))

    # ----- Row 4: shared x-axis label -------------------------------------
    Label(fig[n_rows + 2, 1:n_cols], "Initial vibrational state νᵢ";
          fontsize   = 14,
          tellheight = true,
          tellwidth  = false,
          padding    = (0, 0, 0, 2))

    # ----- Row 5: legend (methods only, colour-coded) ----------------------
    legend_elems  = Any[]
    legend_labels = String[]
    if :arith in methods
        push!(legend_elems,
              [LineElement(; color = COLOR_ARITH, linewidth = 1.8, linestyle = :solid),
               MarkerElement(; color = COLOR_ARITH, marker = :utriangle, markersize = 10,
                               strokecolor = :black, strokewidth = 1.2)])
        push!(legend_labels, "memory (arithmetic)")
    end
    if :endpoint in methods
        push!(legend_elems,
              [LineElement(; color = COLOR_ENDPT, linewidth = 1.8, linestyle = :solid),
               MarkerElement(; color = COLOR_ENDPT, marker = :diamond, markersize = 10,
                               strokecolor = :black, strokewidth = 1.2)])
        push!(legend_labels, "memory (local)")
    end
    if :markovian in methods
        push!(legend_elems,
              [LineElement(; color = COLOR_MARKOV, linewidth = 1.8, linestyle = :solid),
               MarkerElement(; color = COLOR_MARKOV, marker = :circle, markersize = 10,
                               strokecolor = :black, strokewidth = 1.2)])
        push!(legend_labels, "Markovian")
    end

    Legend(fig[n_rows + 3, 1:n_cols],
           legend_elems,
           legend_labels;
           orientation   = :horizontal,
           nbanks        = 1,
           tellwidth     = false,
           tellheight    = true,
           halign        = :center,
           framevisible  = false,
           margin        = (4, 4, 2, 0),
           rowgap        = 2,
           colgap        = 14,
           titlegap      = 4,
           groupgap      = 8,
           patchsize     = (22, 12),
           labelsize     = 12,
           titlesize     = 12)

    # ----- Spacing --------------------------------------------------------
    rowgap!(fig.layout, 1, 0)   # header → panels
    rowgap!(fig.layout, 2, 0)   # vib → tra
    rowgap!(fig.layout, 3, 4)   # tra → x-label
    rowgap!(fig.layout, 4, 2)   # x-label → legend
    # Zero out every column gap so the two data columns sit flush.
    colgap!(fig.layout, 0)

    return fig
end

# ===========================================================================
# NOAu sweep at T = 300 K only — comparing kernel-averaging schemes
# ===========================================================================

const E_TRANS_eV_LIST = [0.2, 0.5, 1.0]     # incident translational energies (eV)
const ν_LIST          = [0, 3, 16]     # EBK quantum numbers

const T_FOCUS = 300

# Mirror DeltaE.jl's DEFAULT_ω_GRID_eV — must stay in sync if changed.
const _ω_GRID_eV = vcat(0.0,
    [1e-7, 3.16e-7, 1e-6, 3.16e-6, 1e-5, 3.16e-5,
     1e-4, 3.16e-4, 1e-3, 3.16e-3],
    collect(0.01:0.01:20.0))

memory_config = Dict{String,Any}(
    "model"          => :NOAu,
    "T_K"            => T_FOCUS,
    "ω"              => _ω_GRID_eV,
    "stride"         => 1,
    "parallel"       => nworkers() > 1,
    # Order = draw order. Arithmetic first → endpoint (local) is drawn
    # last and sits on top in overlapping regions.
    "kernel_average" => [:arithmetic, :endpoint],
)

markovian_config = Dict{String,Any}(
    "model"    => :NOAu,
    "T_K"      => T_FOCUS,
    "stride"   => 1,
    "parallel" => false,
)

# Expand `kernel_average` list into one config per averaging scheme.
configs_kernel = vcat(
    [merge(memory_config, Dict("kernel_average" => ka))
     for ka in memory_config["kernel_average"]],
    markovian_config,
)

# Per-row y-limits — aligned to rows of the 2×2 grid.
panel_ylims = [
    (-0.1, 1.6),       # row 1 — vibrational
    (-0.025, 0.22),   # row 2 — translational
]

fig = plot_vib_DeltaE_kernel(ν_LIST, E_TRANS_eV_LIST, configs_kernel;
                             ylims = panel_ylims)
display(fig)

# Optional save — uncomment to render to disk.
save(plotsdir("cpa", "NOAu", "DeltaE_vs_vib_kernel_compare_SI.pdf"), fig)
