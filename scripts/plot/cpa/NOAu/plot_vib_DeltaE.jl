using Distributed
using DrWatson
@quickactivate "HokseonReproduce"

using Unitful, UnitfulAtomic
using LinearAlgebra: eigen
using StaticArrays: SA
using Random
using HDF5
using HokseonAssistant
using HokseonPlots
using CairoMakie
using ColorSchemes
using Colors

# Sequential temperature colour ramp from HokseonPlots' NICECOLORS — picks N
# evenly spaced colours so cool→warm reads as low-T→high-T.
function temperature_palette(Ts)
    n = length(Ts)
    n == 1 ? [HokseonPlots.NICECOLORS[end-1]] :
             [HokseonPlots.NICECOLORS[Int(round(1 + (length(HokseonPlots.NICECOLORS)-1) * (i-1)/(n-1)))]
              for i in 1:n]
end

HokseonAssistant.julia_build_procs()
@everywhere using HokseonReproduce
@everywhere using StaticArrays: SA
@everywhere using Unitful, UnitfulAtomic

# ---------------------------------------------------------------------------
# Variant detection mirrors CPA_dict_to_data_savename: memory CPA carries
# `kernel_average`, Markovian does not.
# ---------------------------------------------------------------------------
variant_of(cfg) = haskey(cfg, "kernel_average") ? :memory : :markovian
variant_label(::Val{:memory})    = "memory CPA"
variant_label(::Val{:markovian}) = "Markovian CPA"

# ---------------------------------------------------------------------------
# Read one (params_list, cfg) sweep into per-DOF (ν, ΔE_mean, ΔE_lo, ΔE_hi)
# vectors. NOAu CPA writes a 2 × n_traj matrix per file with rows
#   row 1 = ΔE_r (NO bond-stretch, *vibrational* loss),
#   row 2 = ΔE_z (NO–surface separation, *translational* loss).
# We report ensemble means with the 2.5–97.5 percentile interval (95% interval
# of the trajectory distribution, not the CI of the mean — the latter is
# ±1.96·std/√1000 ≈ a few meV, invisible at this scale). Asymmetric (lo, hi)
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
        # ΔE_au_mat may be 2 × n_traj (per-DOF) or, for legacy single-DOF
        # files, a 1 × n_traj / length-n_traj vector. Skip the latter — this
        # plot specifically compares vibrational vs translational, so a file
        # without that decomposition cannot contribute.
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

# Local mean/quantile to avoid a Statistics dependency. quantile uses linear
# interpolation between order statistics (matches Julia stdlib's default).
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
# Local clone of HokseonPlots.MyAxis with a `mirror_top` switch (lifted from
# plot_Et_Γ_DeltaE.jl). MyAxis adds a secondary Axis at xaxisposition=:top to
# mirror x-ticks on the top spine; for stacked panels we want the topmost
# panel to drop those top mirror ticks so the legend can sit cleanly against
# the upper edge.
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
# Render one (x, ΔE) curve into `ax` using the project's variant-aware style:
# memory  — solid line + filled marker + errorbars
# Markov. — dashed line + open marker  + errorbars
# x is the *plot* coordinate (integer index for equal spacing), not ν itself.
# ---------------------------------------------------------------------------
function draw_curve!(ax, xs, ys, lo, hi, col, variant)
    if variant == :memory
        lines!(ax, xs, ys;
               color = col, linewidth = 1.8, linestyle = :solid)
        errorbars!(ax, xs, ys, lo, hi;
                   color = col, whiskerwidth = 8, linewidth = 1.0)
        scatter!(ax, xs, ys;
                 color = col, marker = :circle, markersize = 9,
                 strokecolor = col, strokewidth = 0.6)
    else
        lines!(ax, xs, ys;
               color = col, linewidth = 1.8, linestyle = :dash)
        errorbars!(ax, xs, ys, lo, hi;
                   color = col, whiskerwidth = 8, linewidth = 1.0)
        scatter!(ax, xs, ys;
                 color = :white, marker = :circle, markersize = 9,
                 strokecolor = col, strokewidth = 1.5)
    end
end

# ---------------------------------------------------------------------------
# Build a NOAu params_list for a single incident translational energy. Lifts
# the per-Eₜ params dict to a function (mirrors build_params_list in
# plot_Et_Γ_DeltaE.jl) so each column of the 2×2 figure can be assembled
# with one call. Everything except `translational_kinetic` matches the MD
# sweep so CPA_dict_to_data_savename round-trips to the saved .h5 path.
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
# 2×2 small-multiples publication figure: ΔE vs ν at multiple Eₜ values.
#
# Layout (rows × cols)
#   row 1 — header band: ["Eₜ = 0.5 eV"   [LEGEND]   "Eₜ = 1.0 eV"]  (single
#           row stacked on top of the panels — the Eₜ tags flank the centred
#           legend box, so each tag sits over its own panel column)
#   row 2 — vibrational  panels (a, b)        ← linked y across this rowo
#   row 3 — translational panels (c, d)       ← linked y across this row
#   row 4 — shared x-axis label "ν / EBK vibrational state" (single label
#           spanning both columns, replaces the per-panel xlabel)
#
# Conventions
#   • Columns are ordered **ascending Eₜ** left→right (reads with the eye).
#   • Panels are labelled **row-major** a/b/c/d (top-left, top-right,
#     bottom-left, bottom-right) — matches journal caption style.
#   • Eₜ identification is a *column header* (clean: lives outside the plot
#     box, never competes with data) instead of an in-axis text tag.
#   • y-axis is linked per row: ΔE_r at every Eₜ shares one scale; same for
#     ΔE_z. Direct visual comparison across Eₜ is the whole point of the
#     small-multiples grid.
#   • y-tick *labels* and y-axis labels are shown only on the leftmost
#     column; tick MARKS still live on the inner spines of right-column
#     panels (publication look for shared-y small multiples).
#   • Equal-spacing for ν: x-coordinates are the integer indices
#     1:length(ν_LIST); ticks are labelled with the underlying ν values so
#     the {0, 3, 16} spacing doesn't stretch the high-ν side.
#
# Styling (panel_spinewidth, legend frame, zero row/col gaps) mirrors
# plot_Et_Γ_DeltaE.jl so the NOAu and ET figures read as a set.
# ---------------------------------------------------------------------------
function plot_vib_DeltaE(ν_LIST, E_TRANS_eV_LIST, configs::AbstractVector;
                         figsize          = (HokseonPlots.RESOLUTION[1] * 2.5,
                                             HokseonPlots.RESOLUTION[2] * 3.0),
                         index_label_pos  = (0.03, 0.92),   # (a/b/c/d) tag (top-left)
                         panel_spinewidth = 1.8,
                         ylims            = nothing,        # Vector of (lo, hi) tuples or `nothing` per **row**
                         column_title_size = 16)

    Ts        = sort(unique(c["T_K"] for c in configs))
    color_map = Dict(Ts .=> temperature_palette(Ts))
    variants  = unique(variant_of.(configs))

    n_ticks = length(ν_LIST)
    xticks  = (collect(1:n_ticks), string.(ν_LIST))
    ν_to_x  = Dict(ν_LIST[i] => i for i in 1:n_ticks)

    panel_keys   = (:vib, :tra)                             # top row, bottom row
    panel_ylabels = ("ΔE_r / eV",
                     "ΔE_z / eV")
    n_rows = length(panel_keys)
    n_cols = length(E_TRANS_eV_LIST)

    fig = Figure(; size = figsize,
                   figure_padding = (8, 12, 6, 6),
                   fonts = (; regular = projectdir("fonts", "MinionPro-Capt.otf")))

    # ----- Row 1: header band — Eₜ tags flanking the centred legend -------
    # Eₜ Labels are pinned to the OUTER edges of their column cells
    # (left-most label → halign=:left, right-most → halign=:right) so they
    # sit against the figure's left/right margins, well clear of the
    # centred legend. With halign=:center, the tags landed at each panel
    # column's centre — directly under the (also-centred) legend frame and
    # got blocked by it. The Legend is then added to the same row spanning
    # both columns with halign=:center; all three pieces declare
    # tellwidth=false so they share the row without pushing each other.
    # Row height is dictated by the (taller) legend.
    for (c, Eₜ) in enumerate(E_TRANS_eV_LIST)
        is_first  = c == 1
        is_last   = c == n_cols
        halign_   = is_first ? :left  : (is_last ? :right : :center)
        # Tiny inward padding so the tag aligns with the panel's plot box
        # edge rather than spilling against the figure's outer padding.
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
                ylabel             = is_left ? panel_ylabels[r] : "",
                xminorticksvisible = false,
                yminorticksvisible = true,
                xminorgridvisible  = false,
                yminorgridvisible  = false,
                xgridvisible       = false,
                ygridvisible       = false,
                xticks             = xticks,
                # Per-panel xlabel left empty — replaced by a single shared
                # Label below the bottom row that spans both columns.
                xlabel             = "",
                limits             = (0.5, n_ticks + 0.5, ylo, yhi),
                spinewidth         = panel_spinewidth,
                xtickwidth         = panel_spinewidth,
                ytickwidth         = panel_spinewidth,
                xminortickwidth    = panel_spinewidth,
                yminortickwidth    = panel_spinewidth,
                # Inward ticks on top row, outward on bottom row's x-ticks
                # (publication-standard look). y-ticks stay inward.
                xtickalign         = is_bottom ? 0 : 1,
                xminortickalign    = is_bottom ? 0 : 1,
                ytickalign         = 1,
                yminortickalign    = 1,
            )
            # Layout rows: 1 = header band (Eₜ tags + legend in one row),
            # 2..1+n_rows = panels, n_rows+2 = shared x-axis label.
            ax = panel_axis(fig[r + 1, c]; mirror_top = false, axis_kwargs...)

            for cfg in configs
                data = load_vib_DeltaE(params_list, cfg)
                isempty(data.ν) && continue
                μ, lo, hi = getfield(data, panel_keys[r])
                keep = [haskey(ν_to_x, ν) for ν in data.ν]
                xs   = [ν_to_x[ν] for ν in data.ν[keep]]
                μ_   = μ[keep]; lo_ = lo[keep]; hi_ = hi[keep]
                isempty(xs) && continue
                draw_curve!(ax, xs, μ_, lo_, hi_, color_map[cfg["T_K"]], variant_of(cfg))
            end

            # Row-major panel index: (1,1)=a, (1,2)=b, (2,1)=c, (2,2)=d.
            letter = Char('a' + (r - 1) * n_cols + (c - 1))
            text!(ax, index_label_pos[1], index_label_pos[2];
                  text     = string(letter),
                  space    = :relative,
                  align    = (:left, :top),
                  fontsize = 18,
                  font     = :bold)

            axes[r, c] = ax
        end
    end

    # ----- Linking ---------------------------------------------------------
    # Per-row y-link → ΔE_r (row 1) and ΔE_z (row 2) each share one scale
    # across columns; per-column x-link keeps ν aligned (already identical
    # by construction, but propagates pan/zoom).
    for r in 1:n_rows
        linkyaxes!(axes[r, :]...)
    end
    for c in 1:n_cols
        linkxaxes!(axes[:, c]...)
    end

    # Hide x decorations (tick labels, axis label) on upper rows; keep tick
    # marks visible so the inner row boundary still shows them.
    for r in 1:(n_rows - 1), c in 1:n_cols
        hidexdecorations!(axes[r, c]; grid = false, ticks = false, minorticks = false)
    end
    # Hide y decorations on right-column panels; tick marks remain.
    for r in 1:n_rows, c in 2:n_cols
        hideydecorations!(axes[r, c]; grid = false, ticks = false, minorticks = false)
    end

    # ----- Row 1: shared legend (spans both columns, centred between the
    # two Eₜ tags placed earlier in this same row) -------------------------
    T_elems = [[LineElement(; color = color_map[T], linewidth = 1.8),
                MarkerElement(; color = color_map[T], marker = :circle,
                                markersize = 9, strokecolor = color_map[T],
                                strokewidth = 0.6)] for T in Ts]
    T_labels = ["$(T) K" for T in Ts]

    variant_elems  = Any[]
    variant_labels = String[]
    if :memory in variants
        push!(variant_elems,
              [LineElement(; color = :black, linewidth = 1.8, linestyle = :solid),
               MarkerElement(; color = :black, marker = :circle, markersize = 9,
                               strokecolor = :black, strokewidth = 0.6)])
        push!(variant_labels, "memory")
    end
    if :markovian in variants
        push!(variant_elems,
              [LineElement(; color = :black, linewidth = 1.8, linestyle = :dash),
               MarkerElement(; color = :white, marker = :circle, markersize = 9,
                               strokecolor = :black, strokewidth = 1.5)])
        push!(variant_labels, "Markovian")
    end

    nbanks = max(length(variant_elems), length(T_elems))

    Legend(fig[1, 1:n_cols],
           [variant_elems, T_elems],
           [variant_labels, T_labels],
           ["CPA friction", "temperature"];
           orientation   = :vertical,
           nbanks        = nbanks,
           tellwidth     = false,
           tellheight    = true,
           halign        = :center,        # centred between the two Eₜ tags
           framevisible  = true,
           framecolor    = :black,
           framewidth    = panel_spinewidth,
           # Bottom = 0 so legend frame sits flush on row-2 panels' top spine.
           margin        = (4, 4, 0, 2),
           rowgap        = 2,
           colgap        = 10,
           titlegap      = 4,
           groupgap      = 8,
           patchsize     = (18, 12),
           labelsize     = 11,
           titlesize     = 12,
           titleposition = :left)

    # ----- Row 4: shared x-axis label -------------------------------------
    # Single label spanning both columns replaces the per-panel xlabel that
    # would otherwise duplicate underneath each bottom panel.
    Label(fig[n_rows + 2, 1:n_cols], "ν / EBK vibrational state";
          fontsize   = 14,
          tellheight = true,
          tellwidth  = false,
          padding    = (0, 0, 0, 2))

    # ----- Spacing --------------------------------------------------------
    # Row 1 (header band) flush on top-row panels (row 2); top→bottom panel
    # rows continuous; small gap between bottom panels (row 3) and the
    # shared x-axis label (row 4) so the tick labels and the axis label
    # don't crowd each other.
    rowgap!(fig.layout, 1, 0)
    rowgap!(fig.layout, 2, 0)
    rowgap!(fig.layout, 3, 4)
    # Column gap: 0 so the inner column boundary is a single shared spine
    # carrying both the left panel's y-mirror ticks and the right panel's
    # y-ticks (inward) — the small-multiples publication look.
    for c in 1:(n_cols - 1)
        colgap!(fig.layout, c, 0)
    end

    return fig
end

# ===========================================================================
# NOAu sweep at fixed Eₜ, four temperatures, three vibrational states
# ===========================================================================

const E_TRANS_eV_LIST = [0.2, 1.0]     # incident translational energies (eV)
                                       # — ascending order ⇒ left→right
const ν_LIST          = [0, 3, 16]     # EBK quantum numbers — also x-tick labels

# Three temperatures with both variants → 6 curves max per panel (some memory
# points may be missing — load_vib_DeltaE warns and skips).
const T_K_LIST = [300, 1000, 2000]

# Mirror DeltaE.jl's DEFAULT_ω_GRID_eV — must stay in sync if changed.
# The "ω" field is dropped from CPA_dict_to_data_savename's allowlist, so it
# only documents intent; this script loads from saved files and never calls Λ.
const _ω_GRID_eV = vcat(0.0,
    [1e-7, 3.16e-7, 1e-6, 3.16e-6, 1e-5, 3.16e-5,
     1e-4, 3.16e-4, 1e-3, 3.16e-3],
    collect(0.01:0.01:20.0))

memory_configs = [Dict{String,Any}(
        "model"          => :NOAu,
        "T_K"            => T,
        "ω"              => _ω_GRID_eV,
        "stride"         => 1,
        "parallel"       => nworkers() > 1,
        "kernel_average" => :arithmetic,
        # NB: no `variant` key — the 2-DOF (ΔE_r, ΔE_z) data the new run_memory
        # writer produces lives in `cpa/NOAu/memory/`. Setting
        # `variant => "memory_old"` would route paths to `cpa/NOAu/memory_old/`,
        # where files are the legacy single-DOF (1000,) total-ΔE arrays and
        # would all get skipped by load_vib_DeltaE.
    ) for T in T_K_LIST]

markovian_configs = [Dict{String,Any}(
        "model"   => :NOAu,
        "T_K"     => T,
        "stride"  => 1,
        "parallel" => false,
    ) for T in T_K_LIST]

configs_noau = vcat(memory_configs, markovian_configs)

# Per-row y-limits — `nothing` keeps auto-scaling. Aligned to rows of the
# 2×2 grid: row 1 = vibrational (ΔE_r), row 2 = translational (ΔE_z). Both
# columns share each row's scale via linkyaxes!.
panel_ylims = [
    (-0.1, 1.6),       # row 1 — vibrational
    (nothing, 0.25),   # row 2 — translational
]

fig = plot_vib_DeltaE(ν_LIST, E_TRANS_eV_LIST, configs_noau;
                      ylims = panel_ylims)
display(fig)

# Optional save — uncomment to render to disk.
# save_figure(plotsdir("cpa", "NOAu", "DeltaE_vs_vib_T_variant"), fig)
