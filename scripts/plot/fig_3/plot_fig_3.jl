using DrWatson
@quickactivate "MemoryElectronicFriction"

using HDF5
using CairoMakie
using Colors
using LinearAlgebra: eigmin, Symmetric

# -----------------------------------------------------------------------------
# Reproduce Figure 3 from the data saved by
# scripts/plot/friction/memory/NOAu/plot_LambdaElement_r.jl
# -----------------------------------------------------------------------------

# --- styling constants (kept identical to the original plotting script) ---
const colors = [
    colorant"#1f77b4",
    colorant"#2ca02c",
    colorant"#d62728",
]

const FONT = projectdir("fonts", "dejavu-sans.book.ttf")

const FIG_FRACTION = 1.0
scaled(x) = x .* FIG_FRACTION

const FIGURE_WIDTH = 996
const FIG_WIDTH  = round(Int, FIG_FRACTION * FIGURE_WIDTH)
const FIG_HEIGHT = round(Int, FIG_WIDTH * 0.55)

const panel_spinewidth = scaled(1.8)
const LINE_WIDTH       = scaled(2.5)
const MARKER_SIZE      = scaled(12)
const PANEL_COLGAP     = scaled(10)
const FIG_COLGAP       = scaled(4)

const AXIS_LABEL_SIZE   = scaled(18)
const TITLE_SIZE        = scaled(16)
const TICK_LABEL_SIZE   = scaled(16)
const ANNOTATION_SIZE   = scaled(16)
const LEGEND_LABEL_SIZE = scaled(16)
const LEGEND_TITLE_SIZE = scaled(18)

# --- column reducers cannot be serialized; recreate them by index ---
const COLUMN_REDUCERS = [
    M -> M[1, 1],                # Λ₁₁ (r,r)
    M -> M[2, 2],                # Λ₂₂ (z,z)
    M -> M[1, 2],                # Λ₁₂ (r,z)
    M -> eigmin(Symmetric(M)),   # smallest eigenvalue (least-damped direction)
]

# --- load the reproduction data ---
h5_path = projectdir("figure_data", "fig_3", "fig_3_data.h5")

h5open(h5_path, "r") do h5
    R_VALUES    = read(h5["R_VALUES"])
    Z_VALUES    = read(h5["Z_VALUES"])
    ENERGY_GRID = read(h5["ENERGY_GRID"])
    COLUMN_TITLES = read(h5["COLUMN_TITLES"])
    COLUMN_SCALE  = read(h5["COLUMN_SCALE"])
    unit_conv     = read(h5["unit_conv"])

    N_ROWS = length(R_VALUES)
    N_COLS = length(COLUMN_TITLES)

    fig = Figure(size = (FIG_WIDTH, FIG_HEIGHT),
                 figure_padding = (2, 6, 2, 2),
                 fonts = (; regular = FONT))

    panel_grid = GridLayout(fig[1, 2])
    axes = Matrix{Any}(undef, N_ROWS, N_COLS)

    # --- build panels ---
    for row in 1:N_ROWS, col in 1:N_COLS
        is_bottom = (row == N_ROWS)
        is_top    = (row == 1)
        is_lmin   = (col == N_COLS)
        axes[row, col] = Axis(panel_grid[row, col];
                    title              = is_top ? COLUMN_TITLES[col] : "",
                    titlesize          = TITLE_SIZE,
                    titlefont          = FONT,
                    xlabel             = "",
                    ylabel             = "",
                    xticklabelsize     = TICK_LABEL_SIZE,
                    yticklabelsize     = TICK_LABEL_SIZE,
                    xticklabelsvisible = is_bottom,
                    xscale             = is_lmin ? log10 : identity,
                    limits             = is_lmin ? (0.008, nothing, nothing, nothing) :
                                                   (0.0,   nothing, nothing, nothing),
                    yautolimitmargin   = (0.05, 0.28),
                    xgridvisible       = false, ygridvisible = false,
                    xticksmirrored     = false,
                    yticksmirrored     = true,
                    xtickalign         = 1,
                    ytickalign         = 1,
                    xminortickalign    = 1,
                    yminortickalign    = 1,
                    spinewidth         = panel_spinewidth,
                    xtickwidth         = panel_spinewidth,
                    ytickwidth         = panel_spinewidth)
    end

    linkxaxes!(vec(axes[:, 1:N_COLS-1])...)
    linkxaxes!(axes[:, N_COLS]...)
    rowgap!(panel_grid, scaled(12))
    colgap!(panel_grid, PANEL_COLGAP)

    # λ_min-column threshold-energy markers, collected here and drawn in a SECOND
    # pass so each × sits on top of all the λ_min curves in its panel.
    threshold_markers = Tuple{Int, Float64, eltype(colors)}[]

    # --- first pass: draw curves and dashed Markovian lines ---
    for (row, r) in enumerate(R_VALUES)
        zs = sort(Z_VALUES)
        for (i, z) in enumerate(zs)
            r_group = h5["r_$(r)"]
            z_group = r_group["z_$(z)"]

            Lambda_mats = read(z_group["Lambda_mats"])   # (N_ω, 2, 2)
            gamma_mat   = read(z_group["gamma_mat"])     # (2, 2)
            threshold_energy_eV = only(read(z_group["threshold_energy_eV"]))

            push!(threshold_markers, (row, threshold_energy_eV, colors[i]))

            Nω = size(Lambda_mats, 1)
            for col in 1:N_COLS
                reduce_fn = COLUMN_REDUCERS[col]
                scale = unit_conv * COLUMN_SCALE[col]

                curve = [reduce_fn(Lambda_mats[j, :, :]) for j in 1:Nω] .* scale
                mark  = reduce_fn(gamma_mat) * scale

                lines!(axes[row, col], ENERGY_GRID, curve;
                       color = colors[i], linestyle = :solid, linewidth = LINE_WIDTH)
                hlines!(axes[row, col], [mark];
                        color = colors[i], linestyle = :dash, linewidth = LINE_WIDTH)
            end
        end

        # Per-row fixed-r label on the right margin.
        Label(panel_grid[row, N_COLS + 1], "r = $r Å";
              rotation = π/2, tellheight = false, tellwidth = true,
              fontsize = TITLE_SIZE, padding = scaled((4, 0, 0, 0)))
    end

    colgap!(panel_grid, PANEL_COLGAP)

    # --- second pass: threshold-energy × markers on top of λ_min curves ---
    for (row, x, c) in threshold_markers
        scatter!(axes[row, N_COLS], [x], [0];
                 color = c, marker = :x, markersize = MARKER_SIZE)
    end

    # --- panel labels (a)-(h), row-major ---
    for row in 1:N_ROWS, col in 1:N_COLS
        letter = 'a' + (row - 1) * N_COLS + (col - 1)
        text!(axes[row, col], 0.04, 0.95; text = "($(letter))", space = :relative,
              align = (:left, :top), fontsize = ANNOTATION_SIZE,
              strokewidth = scaled(0.4), strokecolor = :black)
    end

    # --- shared axis labels ---
    Label(fig[1, 1], "Friction  (u⋅ps⁻¹)";
          rotation = π/2, tellheight = false, tellwidth = true,
          fontsize = AXIS_LABEL_SIZE, padding = (0, 0, 0, 0))
    Label(fig[2, 2], "ħω  (eV)";
          tellwidth = false, tellheight = true,
          fontsize = AXIS_LABEL_SIZE, padding = (0, 0, 2, 0))

    # --- top horizontal three-group legend ---
    zs_sorted = sort(Z_VALUES)
    color_elems  = [LineElement(color = colors[i], linewidth = LINE_WIDTH)
                    for i in 1:length(zs_sorted)]
    color_labels = ["$z" for z in zs_sorted]

    style_elems  = [LineElement(color = :black, linestyle = :solid, linewidth = LINE_WIDTH),
                    LineElement(color = :black, linestyle = :dash,  linewidth = LINE_WIDTH)]
    style_labels = ["Memory", "Markovian"]

    threshold_elems  = [MarkerElement(marker = :x, color = :black, markersize = MARKER_SIZE)]
    threshold_labels = ["ħω*"]

    legend_kwargs = (
        orientation   = :horizontal,
        titleposition = :left,
        nbanks        = 1,
        tellwidth     = false,
        tellheight    = true,
        halign        = :center,
        framevisible  = false,
        patchsize     = scaled((18, 10)),
        colgap        = scaled(5),
        groupgap      = scaled(0),
        titlegap      = scaled(4),
        patchlabelgap = scaled(3),
        padding       = scaled((2, 2, 1, 1)),
        labelsize     = LEGEND_LABEL_SIZE,
        titlesize     = LEGEND_TITLE_SIZE,
        labelfont     = FONT,
        titlefont     = FONT,
    )

    Legend(fig[0, 1:2],
           [color_elems, style_elems, threshold_elems],
           [color_labels, style_labels, threshold_labels],
           ["z  (Å)", "   |   Friction", "   |   PSD threshold"];
           legend_kwargs...)

    colgap!(fig.layout, FIG_COLGAP)
    rowgap!(fig.layout, scaled(2))

    display(fig)

    outpath = plotsdir("fig_3", "fig_3.pdf")
    mkpath(dirname(outpath))
    save(outpath, fig)
end
