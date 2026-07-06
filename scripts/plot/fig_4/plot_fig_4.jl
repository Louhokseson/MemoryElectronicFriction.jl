using DrWatson
@quickactivate "HokseonReproduce"

using HDF5
using CairoMakie
using Colors
using ColorSchemes

# -----------------------------------------------------------------------------
# Reproduce Figure 4 from the data saved by
# scripts/plot/cpa/ErpenbeckThoss/plot_Γ_DeltaE.jl
# -----------------------------------------------------------------------------

# --- styling constants (kept identical to the original plotting script) ---
const TWO_COLUMN_WIDTH = 481
const FIG_WIDTH         = TWO_COLUMN_WIDTH
const FIG_HEIGHT        = round(Int, FIG_WIDTH * 0.6)
const AXIS_LABEL_SIZE   = 18
const TICK_LABEL_SIZE   = 16
const LEGEND_LABEL_SIZE = 16
const LEGEND_TITLE_SIZE = 16
const ANNOTATION_SIZE   = 16
const LINE_WIDTH        = 1.8
const MARKER_SIZE       = 12
const SPINE_WIDTH       = 1.8

const FONT = projectdir("fonts", "dejavu-sans.book.ttf")

# Sequential temperature colour ramp from HokseonPlots' NICECOLORS.
const NICECOLORS = ColorScheme(parse.(Colorant, ["#FCE1A4","#FABF7B","#F08F6E","#E05C5C","#D12959","#AB1866","#6E005F"]))
function temperature_palette(Ts)
    n = length(Ts)
    n == 1 ? [NICECOLORS[end-1]] :
             [NICECOLORS[Int(round(1 + (length(NICECOLORS)-1) * (i-1)/(n-1)))]
              for i in 1:n]
end

# Per-variant curve styling.
variant_label(::Val{:memory_arithmetic}) = "Memory (arithmetic)"
variant_label(::Val{:memory_local})      = "Memory"
variant_label(::Val{:markovian})         = "Markovian"

variant_style(::Val{:memory_arithmetic}) = (linestyle = :solid, marker = :diamond,  filled = true)
variant_style(::Val{:memory_local})      = (linestyle = :solid, marker = :circle, filled = true)
variant_style(::Val{:markovian})         = (linestyle = :dash,  marker = :circle,  filled = false)

const VARIANT_ORDER = (:memory_arithmetic, :memory_local, :markovian)

# -----------------------------------------------------------------------------
# Load the reproduction data
# -----------------------------------------------------------------------------
h5_path = projectdir("figure_data", "fig_4", "fig_4_data.h5")

h5open(h5_path, "r") do h5
    Gamma_values = read(h5["Gamma_values"])
    Delta0_values = read(h5["Delta0_values"])
    E_trans_eV  = read(h5["E_trans_eV"])
    T_K_values  = sort(read(h5["T_K_values"]))
    model       = read(h5["model"])

    color_map = Dict(T_K_values .=> temperature_palette(T_K_values))

    curves = NamedTuple[]
    curves_group = h5["curves"]
    for k in sort!(collect(keys(curves_group)))
        g = curves_group[k]
        ΔE  = read(g["DeltaE_eV"])
        T_K = read(g["T_K"])
        var = read(g["variant"])
        vlab = read(g["variant_label"])
        valid = Bool.(read(g["valid"]))
        push!(curves, (;
            DeltaE_eV    = ΔE,
            T_K          = T_K,
            variant      = Symbol(var),
            variant_label = vlab,
            valid        = valid,
        ))
    end

    variants = unique(c.variant for c in curves)

    # --- figure and axis ---
    fig = Figure(; size = (FIG_WIDTH, FIG_HEIGHT),
                   figure_padding = (8, 12, 6, 6),
                   fonts = (; regular = FONT))

    axis_kwargs = (
        xlabel = "Δ₀ (eV)",
        ylabel = "ΔE (eV)",
        xscale = log10,
        xlabelsize     = AXIS_LABEL_SIZE,
        ylabelsize     = AXIS_LABEL_SIZE,
        xticklabelsize = TICK_LABEL_SIZE,
        yticklabelsize = TICK_LABEL_SIZE,
        xminorticksvisible = false,
        yminorticksvisible = true,
        xminorgridvisible  = false,
        yminorgridvisible  = false,
        xgridvisible       = false,
        ygridvisible       = false,
        xtickalign = 1, ytickalign = 1,
        xminortickalign = 1, yminortickalign = 1,
        spinewidth = SPINE_WIDTH,
        xtickwidth = SPINE_WIDTH, ytickwidth = SPINE_WIDTH,
        # Turn on top and right ticks
        xticksmirrored = true,
        yticksmirrored = true,
    )
    ax = Axis(fig[1, 1]; axis_kwargs..., xticks = (Delta0_values, string.(Delta0_values)))

    # --- draw curves ---
    for c in curves
        any(c.valid) || continue
        Δ0 = Delta0_values[c.valid]
        ΔE = c.DeltaE_eV[c.valid]
        col = color_map[c.T_K]
        st  = variant_style(Val(c.variant))
        lines!(ax, Δ0, ΔE;
               color = col, linewidth = LINE_WIDTH, linestyle = st.linestyle)
        scatter!(ax, Δ0, ΔE;
                 color       = st.filled ? col : :white,
                 marker      = st.marker, markersize = MARKER_SIZE,
                 strokecolor = col, strokewidth = st.filled ? 0.5 : 1.0)
    end

    # --- legend ---
    T_elems = [[LineElement(; color = color_map[T], linewidth = LINE_WIDTH),
                MarkerElement(; color = color_map[T], marker = :circle,
                                markersize = MARKER_SIZE, strokecolor = color_map[T],
                                strokewidth = 0.5)] for T in T_K_values]
    T_labels = ["$(T) K" for T in T_K_values]

    vcol = length(T_K_values) == 1 ? color_map[only(T_K_values)] : :black
    variant_elems = Any[]
    variant_labels = String[]
    for v in VARIANT_ORDER
        v in variants || continue
        st = variant_style(Val(v))
        push!(variant_elems,
              [LineElement(; color = vcol, linewidth = LINE_WIDTH, linestyle = st.linestyle),
               MarkerElement(; color = st.filled ? vcol : :white, marker = st.marker,
                               markersize = MARKER_SIZE, strokecolor = vcol,
                               strokewidth = st.filled ? 0.5 : 1.0)])
        push!(variant_labels, variant_label(Val(v)))
    end

    groups       = length(T_K_values) > 1 ? [T_elems, variant_elems]   : [variant_elems]
    group_labels = length(T_K_values) > 1 ? [T_labels, variant_labels] : [variant_labels]
    group_titles = length(T_K_values) > 1 ? ["temperature", "CPA"]     : [nothing]

    Legend(fig[1, 1],
           groups, group_labels, group_titles;
           tellwidth   = false, tellheight = false,
           valign      = :top,
           halign      = :right,
           margin      = (4, 4, 4, 4),
           framevisible = false,
           framecolor   = :black,
           framewidth   = 1.2,
           rowgap       = 1,
           titlegap     = 3,
           groupgap     = 6,
           patchsize    = (30, 9),
           labelsize    = LEGEND_LABEL_SIZE,
           titlesize    = LEGEND_TITLE_SIZE)

    display(fig)

    outpath = plotsdir("fig_4", "fig_4.pdf")
    mkpath(dirname(outpath))
    save(outpath, fig)
end
