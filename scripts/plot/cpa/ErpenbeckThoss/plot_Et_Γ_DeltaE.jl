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
# Read one (params_list, cfg) sweep into (Γ, ΔE) vectors. Skips and warns on
# missing files so a partially-finished sweep still plots cleanly.
# ---------------------------------------------------------------------------
function load_Γ_DeltaE(params_list, cfg)
    Γ_eV      = Float64[]
    DeltaE_eV = Float64[]
    for p in params_list
        savingpath, savingname = CPA_dict_to_data_savename(p, cfg)
        full_data_path = datadir(savingpath, savingname)
        if !isfile(full_data_path)
            @warn "Missing CPA data file" full_data_path; continue
        end
        ΔE_au = h5open(full_data_path, "r") do f
            read(f, "DeltaE_au")
        end
        push!(Γ_eV,      ustrip(u"eV", p["Γ"]))
        push!(DeltaE_eV, ustrip(auconvert(u"eV", first(ΔE_au))))
    end
    order = sortperm(Γ_eV)
    return Γ_eV[order], DeltaE_eV[order]
end

# ---------------------------------------------------------------------------
# Build the per-Eₜ params_list. Identical to the single-panel script but the
# translational kinetic energy is taken as an argument so it can be swept.
# ---------------------------------------------------------------------------
function build_params_list(E_TRANS_eV, Γ_eV_LIST)
    all_params = Dict{String, Any}(
        "mass"                  => [10.54u"u"],
        "Γ"                     => Γ_eV_LIST .* u"eV",
        "r0"                    => [[5.0u"Å"]],
        "translational_kinetic" => [E_TRANS_eV * u"eV"],
        "state"                 => [1],
        "tmax"                  => [200.0u"fs"],
        "dt"                    => [0.01u"fs"],
        "termination_min_time"  => [10.0u"fs"],
        "termination_coord_idx" => [1],
        "termination_threshold" => [5.0u"Å"],
    )
    return dict_list(all_params)
end

# ---------------------------------------------------------------------------
# Render one (Γ, ΔE) curve into `ax` using the project's variant-aware style:
# memory  — solid line + filled marker
# Markov. — dashed line + open marker
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Local clone of HokseonPlots.MyAxis with a `mirror_top` switch. MyAxis adds a
# secondary Axis at xaxisposition=:top to mirror x-ticks on the top spine; for
# stacked panels we want the topmost panel to drop those top mirror ticks so
# the legend can sit cleanly against the upper edge.
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

function draw_curve!(ax, Γ_eV, ΔE_eV, col, variant)
    if variant == :memory
        lines!(ax, Γ_eV, ΔE_eV;
               color = col, linewidth = 1.8, linestyle = :solid)
        scatter!(ax, Γ_eV, ΔE_eV;
                 color = col, marker = :circle, markersize = 9,
                 strokecolor = col, strokewidth = 0.6)
    else
        lines!(ax, Γ_eV, ΔE_eV;
               color = col, linewidth = 1.8, linestyle = :dash)
        scatter!(ax, Γ_eV, ΔE_eV;
                 color = :white, marker = :circle, markersize = 9,
                 strokecolor = col, strokewidth = 1.5)
    end
end

# ---------------------------------------------------------------------------
# Three-panel publication figure: ΔE vs Γ at three incident translational
# energies, stacked vertically with a shared Γ axis. Each panel reuses the
# temperature × CPA-variant encoding; a single legend at the top serves all
# three panels so the in-axis area stays clean.
# ---------------------------------------------------------------------------
function plot_Et_Γ_DeltaE(E_TRANS_eV_list, Γ_eV_LIST, configs::AbstractVector;
                          xscale           = log10,
                          xticks           = (Γ_eV_LIST, string.(Γ_eV_LIST)),
                          figsize          = (HokseonPlots.RESOLUTION[1] * 2.0,
                                              HokseonPlots.RESOLUTION[2] * 4.0),
                          panel_label_pos  = (0.97, 0.92),    # Eₜ tag (top-right)
                          index_label_pos  = (0.03, 0.82),    # (a)/(b)/(c) tag (top-left, lowered)
                          panel_spinewidth = 1.8,
                          ylims            = nothing)         # Vector of (lo, hi) tuples or `nothing` per panel

    Ts        = sort(unique(c["T_K"] for c in configs))
    color_map = Dict(Ts .=> temperature_palette(Ts))
    variants  = unique(variant_of.(configs))
    n_panels  = length(E_TRANS_eV_list)

    fig = Figure(; size = figsize,
                   figure_padding = (8, 12, 6, 6),
                   fonts = (; regular = projectdir("fonts", "MinionPro-Capt.otf")))

    axes = Any[]
    for (i, Eₜ) in enumerate(E_TRANS_eV_list)
        is_bottom = i == n_panels
        # Resolve per-panel y-limits → Makie `limits=(xlo, xhi, ylo, yhi)`
        # tuple. Set at construction time because post-hoc `ylims!` gets
        # clobbered by `linkaxes!(main, extra)` inside `panel_axis` (the
        # mirror axis has no data, so the y-link reconciles to auto).
        ylo, yhi = if ylims === nothing || ylims[i] === nothing
            (nothing, nothing)
        else
            ylims[i]
        end
        axis_kwargs = (
            ylabel = "",                # shared y-label drawn separately
            xscale = xscale,
            xminorticksvisible = false,
            yminorticksvisible = true,
            xminorgridvisible  = false,
            yminorgridvisible  = false,
            xgridvisible       = false,
            ygridvisible       = false,
            xticks             = xticks,
            xlabel             = is_bottom ? "Γ / eV" : "",
            limits             = (nothing, nothing, ylo, yhi),
            # Thicker spines + tick widths for publication-quality framing.
            spinewidth         = panel_spinewidth,
            xtickwidth         = panel_spinewidth,
            ytickwidth         = panel_spinewidth,
            xminortickwidth    = panel_spinewidth,
            yminortickwidth    = panel_spinewidth,
            # Tick alignment: inward (=1) for the upper panels so the inner
            # boundary ticks live in their own plot box; OUTWARD (=0) for
            # the bottom panel's x-ticks, the standard publication look —
            # the outer-edge tick labels read against ticks that protrude
            # below the box. y-ticks stay inward on every panel.
            # HokseonPlots' get_theme sets these, but its theme is only
            # applied inside save_figure's `with_theme`; when displaying
            # the figure directly we set them on the axis explicitly.
            xtickalign         = is_bottom ? 0 : 1,
            xminortickalign    = is_bottom ? 0 : 1,
            ytickalign         = 1,
            yminortickalign    = 1,
        )
        # Row 1 hosts the legend box (sits flush on top of panel 1), so
        # panels occupy rows 2..n_panels+1. ALL panels drop their top mirror
        # ticks: the topmost panel because the legend rests on its top edge,
        # and the rest because the inner boundaries between stacked panels
        # would otherwise show ticks pointing into the neighbour above.
        ax = panel_axis(fig[i + 1, 1]; mirror_top = false, axis_kwargs...)

        params_list = build_params_list(Eₜ, Γ_eV_LIST)
        for cfg in configs
            Γ_eV, ΔE_eV = load_Γ_DeltaE(params_list, cfg)
            isempty(Γ_eV) && continue
            draw_curve!(ax, Γ_eV, ΔE_eV, color_map[cfg["T_K"]], variant_of(cfg))
        end

        # In-axis Eₜ tag (top-right) and panel index "(a)/(b)/(c)" (top-left)
        # — keeps each panel self-identifying without stealing vertical
        # space for a per-axis title.
        text!(ax, panel_label_pos[1], panel_label_pos[2];
              text   = "Eₜ = $(Eₜ) eV",
              space  = :relative,
              align  = (:right, :top),
              fontsize = 16)
        text!(ax, index_label_pos[1], index_label_pos[2];
              text   = "$(Char('a' + i - 1))",
              space  = :relative,
              align  = (:left, :top),
              fontsize = 18,
              font     = :bold)

        push!(axes, ax)
    end

    # Shared Γ axis — pan/zoom in any panel updates the rest.
    linkxaxes!(axes...)
    # Hide tick *labels* and the axis label on the upper panels (so only
    # panel (c) shows Γ values + "Γ / eV"), but keep the tick MARKS visible.
    # With the HokseonPlots theme setting `xtickalign = 1`, those bottom
    # ticks point inward (upward into the panel's own plot box), which is
    # the publication-standard look for stacked-panel shared-x figures.
    for ax in axes[1:end-1]
        hidexdecorations!(ax; grid = false, ticks = false, minorticks = false)
    end

    # Shared y-axis label in column 0, spanning the panel rows (rows 2..n+1
    # since row 1 is the legend). Keeps the figure compact.
    Label(fig[2:(n_panels + 1), 0], "ΔE / eV";
          rotation = π/2,
          fontsize = 14,
          padding  = (0, -6, 0, 0))

    # ----- Single shared legend with two grouped rows -----------------------
    # `orientation = :vertical` stacks the two groups (CPA, temperature) on
    # top of each other inside one frame; `nbanks = max items` lets the items
    # within each group spread horizontally into that many columns, so each
    # group reads as a single horizontal row. CPA listed first => on top.
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

    # Legend sits in its own row (row 1) directly on top of the first panel
    # — `tellheight = true` claims its own row, `tellwidth = false` lets it
    # span the column without forcing its width on the panels below.
    Legend(fig[1, 1],
           [variant_elems, T_elems],
           [variant_labels, T_labels],
           ["CPA friction", "temperature"];
           orientation   = :vertical,
           nbanks        = nbanks,
           tellwidth     = false,
           tellheight    = true,
           framevisible  = true,
           framecolor    = :black,
           framewidth    = panel_spinewidth,
           # Margin order in Makie: (left, right, bottom, top). Bottom = 0
           # so the legend's bottom frame line lands flush on panel (a)'s
           # top spine; rowgap = 0 between cells alone is not enough — the
           # block's own external margin would still leave a visible strip.
           margin        = (4, 4, 0, 2),
           rowgap        = 2,
           colgap        = 10,
           titlegap      = 4,
           groupgap      = 8,
           patchsize     = (18, 12),
           labelsize     = 11,
           titlesize     = 12,
           titleposition = :left)

    # Zero gaps everywhere: legend rests flush on the top panel, and panels
    # share the Γ axis as one continuous block.
    for r in 1:n_panels
        rowgap!(fig.layout, r, 0)
    end

    return fig
end

# ===========================================================================
# Erpenbeck–Thoss sweep at three Eₜ × three temperatures × two CPA variants
# ===========================================================================

const E_TRANS_eV_LIST = [1.0, 2.0, 3.0]                     # incident translational energies (eV)
const Γ_eV_LIST       = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]   # also used as x-tick locations

# Three temperatures with both variants → 6 curves per panel.
const T_K_LIST = [300, 1000, 2000]

# Mirror DeltaE.jl's DEFAULT_ω_GRID_eV — must stay in sync if changed.
# The "ω" field is dropped from CPA_dict_to_data_savename's allowlist, so it
# only documents intent; this script loads from saved files and never calls Λ.
const _ω_GRID_eV = vcat(0.0,
    [1e-7, 3.16e-7, 1e-6, 3.16e-6, 1e-5, 3.16e-5,
     1e-4, 3.16e-4, 1e-3, 3.16e-3],
    collect(0.01:0.01:20.0))

memory_configs = [Dict{String,Any}(
        "model"          => :ErpenbeckThoss,
        "T_K"            => T,
        "ω"              => _ω_GRID_eV,
        "stride"         => 1,
        "parallel"       => nworkers() > 1,
        "kernel_average" => :arithmetic,
    ) for T in T_K_LIST]

markovian_configs = [Dict{String,Any}(
        "model"   => :ErpenbeckThoss,
        "T_K"     => T,
        "stride"  => 1,
        "parallel" => false,
    ) for T in T_K_LIST]

configs_et = vcat(memory_configs, markovian_configs)

# Per-panel y-limits — use `nothing` for an entry to keep auto-scaling, or
# `(lo, hi)` with either side `nothing` to clamp only one bound. Aligned to
# E_TRANS_eV_LIST: index i ↔ panel for E_TRANS_eV_LIST[i].
panel_ylims = [
    (-0.1, 1.2),            # (a) Eₜ = 1.0 eV — auto
    (-3.0, nothing),    # (b) Eₜ = 2.0 eV — clamp lower bound at -3, upper auto
    (nothing, 64),            # (c) Eₜ = 3.0 eV — auto
]

fig = plot_Et_Γ_DeltaE(E_TRANS_eV_LIST, Γ_eV_LIST, configs_et;
                       ylims = panel_ylims)
display(fig)

# Optional save — uncomment to render to disk.
# save_figure(plotsdir("cpa", "ErpenbeckThoss", "DeltaE_vs_Gamma_T_variant_Et_panels"), fig)