using DrWatson
@quickactivate "HokseonReproduce"

using HDF5
using CairoMakie
using Colors

# -----------------------------------------------------------------------------
# Reproduce Figure 2 from the data saved by
# scripts/plot/friction/memory/ErpenbeckThoss/plot_Lambda_Gamma_log_vertical.jl
# -----------------------------------------------------------------------------

# --- styling constants (kept identical to the original plotting script) ---
const TWO_COLUMN_WIDTH = 481
const XLIMITS = (-0.2, 11.0)
const panel_labels = ["(a)", "(b)", "(c)"]
const panel_spinewidth = 1.8
const FONT = projectdir("fonts", "dejavu-sans.book.ttf")

const AXIS_LABEL_SIZE   = 18
const TICK_LABEL_SIZE   = 16
const ANNOTATION_SIZE   = 18
const LEGEND_LABEL_SIZE = 16
const LEGEND_TITLE_SIZE = 18

const colors = [
    colorant"#1f77b4",
    colorant"#2ca02c",
    colorant"#ff7f0e",
    colorant"#d62728",
]

# --- load the reproduction data ---
h5_path = projectdir("figure_data", "fig_2", "fig_2_data.h5")

h5 = h5open(h5_path, "r")
positions   = read(h5["positions"])
Γ_values    = read(h5["Gamma_values"])
C_FERMI     = read(h5["C_FERMI"])
# Keep the legend text identical when C_FERMI was written as an integer (e.g. 3.0 → 3).
C_FERMI_LABEL = isinteger(C_FERMI) ? Int(C_FERMI) : C_FERMI
# temperature, energy_grid and impuritymodel are stored but not needed for plotting
panel_names = ["panel_a", "panel_b", "panel_c"]

fig = Figure(
    size = (TWO_COLUMN_WIDTH, TWO_COLUMN_WIDTH),
    figure_padding = (2, 6, 2, 2),
    fonts = (; regular = FONT),
)

axes = Axis[]

for (k, Γ) in enumerate(Γ_values)
    is_bottom = (k == length(Γ_values))
    panel_group = h5[panel_names[k]]

    # First pass: read curves and compute the panel's log y-window.
    panel_curves = NamedTuple[]
    panel_ymax   = 0.0
    panel_ymin   = Inf

    for (i, pos) in enumerate(positions)
        pos_group = panel_group["position_$(pos)"]
        ω_ev   = read(pos_group["omega_ev"])
        γ      = read(pos_group["gamma"])
        Λ      = read(pos_group["Lambda"])
        ω_peak = only(read(pos_group["omega_peak"]))

        posvals    = filter(x -> isfinite(x) && x > 0, vcat(γ, Λ))
        panel_ymax = max(panel_ymax, maximum(posvals))
        panel_ymin = min(panel_ymin, minimum(posvals))

        push!(panel_curves, (;
            i        = i,
            ω_ev     = ω_ev,
            γ        = γ,
            Λ        = Λ,
            ω_peak   = ω_peak,
        ))
    end

    r_top = 0.72
    α     = (1 - r_top) / r_top
    D_dec = log10(panel_ymax / panel_ymin)
    y_bot = panel_ymin / 1.2
    y_top = panel_ymax * 10.0^(α * D_dec)

    ax = Axis(fig[k, 2];
        xlabel           = is_bottom ? "ħω  (eV)" : "",
        ylabel           = "",
        yscale           = log10,
        xlabelsize       = AXIS_LABEL_SIZE,
        xticklabelsize   = TICK_LABEL_SIZE,
        yticklabelsize   = TICK_LABEL_SIZE,
        limits           = (XLIMITS..., y_bot, y_top),
        xgridvisible     = false,
        ygridvisible     = false,
        xticklabelsvisible = is_bottom,
        xticksvisible    = is_bottom,
        xticks           = 0:2:10,
        xticksmirrored   = false,
        yticksmirrored   = true,
        xtickalign       = 1,
        ytickalign       = 1,
        xminortickalign  = 1,
        yminortickalign  = 1,
        spinewidth       = panel_spinewidth,
        xtickwidth       = panel_spinewidth,
        ytickwidth       = panel_spinewidth,
    )

    if !is_bottom
        hidexdecorations!(ax; ticks = true, ticklabels = true, label = true,
                              minorticks = true, grid = false)
    end

    ax.xticksmirrored[]     = false
    ax.xminorticksvisible[] = false

    for content in contents(fig[k, 2])
        if content !== ax && content isa Axis
            hidexdecorations!(content; ticks = true, minorticks = true,
                              ticklabels = true, label = true, grid = false)
        end
    end

    if k > 1
        ax.topspinevisible[] = false
    end

    push!(axes, ax)

    # Second pass: draw curves and peak markers.
    for c in panel_curves
        lines!(ax, c.ω_ev, c.γ; color = colors[c.i], linestyle = :dash,  linewidth = 2)
        vlines!(ax, c.ω_peak;     color = colors[c.i], linestyle = :dot,   linewidth = 1.5)
        lines!(ax, c.ω_ev, c.Λ;  color = colors[c.i], linestyle = :solid, linewidth = 2)
    end

    # Panel label and Γ annotation.
    text!(ax, 0.98, 0.97; text = panel_labels[k], space = :relative,
          align = (:right, :top), fontsize = ANNOTATION_SIZE,
          strokewidth = 0.6, strokecolor = :black)

    text!(ax, 0.5, 0.95; text = "Δ₀ = $(Γ/2) eV", space = :relative,
          align = (:center, :top), fontsize = ANNOTATION_SIZE)
end

linkxaxes!(axes...)
rowgap!(fig.layout, 0)

# Shared y-label.
Label(fig[1:length(Γ_values), 1], "K(ω ; x)  (u⋅ps⁻¹)";
      rotation = π/2, tellheight = false, tellwidth = true,
      fontsize = AXIS_LABEL_SIZE, padding = (0, 4, 0, 0))

# -----------------------------------------------------------------------------
# Top horizontal legend
# -----------------------------------------------------------------------------
color_elems  = [LineElement(color = colors[i], linewidth = 2) for i in 1:length(positions)]
color_labels = ["$(p)" for p in positions]

style_elems  = [
    LineElement(color = :black, linestyle = :solid, linewidth = 2),
    LineElement(color = :black, linestyle = :dash,  linewidth = 2),
]
style_labels = ["Memory", "Markovian"]

peak_elems  = [LineElement(
    points = [Point2f(0.5, 0), Point2f(0.5, 1)],
    color = :gray25, linestyle = :dot, linewidth = 1.5,
)]
peak_labels = ["ħω = |h(x)| + $(C_FERMI_LABEL) kʙT"]

legend_grid = GridLayout(fig[0, 1:2]; halign = :center, tellwidth = true)

legend_kwargs = (
    orientation   = :horizontal,
    nbanks        = 1,
    tellwidth     = true,
    tellheight    = true,
    halign        = :left,
    framevisible  = false,
    patchsize     = (26, 12),
    colgap        = 8,
    patchlabelgap = 4,
    padding       = (2, 2, 1, 1),
    labelsize     = LEGEND_LABEL_SIZE,
    labelfont     = FONT,
)

legend_titles = ["x  (Å)", "Friction", "Peak"]
for (r, t) in enumerate(legend_titles)
    Label(legend_grid[r, 1], t; halign = :right,
          fontsize = LEGEND_TITLE_SIZE, font = FONT)
end

Legend(legend_grid[1, 2], color_elems, color_labels; legend_kwargs...)
Legend(legend_grid[2, 2], style_elems, style_labels; legend_kwargs...)
Legend(legend_grid[3, 2], peak_elems,  peak_labels;  legend_kwargs...)

colgap!(legend_grid, 8)
rowgap!(legend_grid, 2)

colgap!(fig.layout, 4)

close(h5)

display(fig)

outdir = plotsdir("fig_2")
mkpath(outdir)
save(joinpath(outdir, "fig_2.pdf"), fig)
