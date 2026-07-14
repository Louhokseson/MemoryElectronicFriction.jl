using DrWatson
@quickactivate "MemoryElectronicFriction"

using HDF5
using CairoMakie

# -----------------------------------------------------------------------------
# Reproduce Figure 5 from the data saved by
# scripts/plot/cpa/NOAu/plot_vib_state_distribution_paper.jl
# -----------------------------------------------------------------------------

# --- styling constants (kept identical to the original plotting script) ---
const TWO_COLUMN_WIDTH = 481
const FONT = projectdir("fonts", "dejavu-sans.book.ttf")

const panel_spinewidth = 1.8
const tick_size = 6
const subtick_size = 4
const color_endpt = "#B53A8C"
const color_mark  = "#3F9DCC"
const color_md    = "#444444"

const title_fontsize         = 16
const label_fontsize         = 18
const tick_fontsize          = 16
const legend_fontsize        = 15
const panel_label_fontsize   = 16

# --- row_xspec logic (identical to the original) ---
function row_xspec(ν_init)
    if ν_init ≤ 5
        return (xticks = 0:1:5,         xlims = (-0.5, 5.5))
    elseif ν_init ≤ 10
        return (xticks = 0:2:10,        xlims = (-0.5, 10.5))
    else
        return (xticks = 6:3:18,        xlims = (5, 19))
    end
end

function row_yspec(ν_init)
    if ν_init > 10
        return (yticks = 0:0.2:0.6, ymax = 0.62)
    else
        return (yticks = Makie.automatic, ymax = nothing)
    end
end

# --- load the reproduction data ---
h5_path = projectdir("figure_data", "fig_5", "fig_5_data.h5")

h5 = h5open(h5_path, "r")
nu_initial_list = read(h5["nu_initial_list"])
E_trans_eV_list = read(h5["E_trans_eV_list"])
nu_max = Int(read(h5["nu_max"]))
show_md_baseline = Bool(read(h5["show_md_baseline"]))

# Collect and sort panel names: panel_1_1, panel_1_2, panel_1_3, panel_2_1, ...
panel_names = filter(p -> startswith(p, "panel_"), collect(keys(h5)))
sort!(panel_names, by = p -> begin
    parts = split(p, "_")
    (parse(Int, parts[2]), parse(Int, parts[3]))
end)

n_rows = length(nu_initial_list)
n_cols = length(E_trans_eV_list)

# --- build figure ---
fig = Figure(;
    size = (TWO_COLUMN_WIDTH, TWO_COLUMN_WIDTH),
    figure_padding = (8, 12, 6, 6),
    fonts = (; regular = FONT),
)

# Column headers: Eₜ labels
for (c, Eₜ) in enumerate(E_trans_eV_list)
    Label(fig[1, c], "Eₜ = $(Eₜ) eV";
          fontsize   = title_fontsize,
          halign     = :center,
          tellwidth  = false,
          tellheight = true,
          padding    = (0, 0, 2, 0))
end

axes = Matrix{Any}(undef, n_rows, n_cols)

for pname in panel_names
    panel = h5[pname]
    nu_init       = read(panel["nu_init"])
    panel_letter  = read(panel["panel_letter"])
    nu_bins       = read(panel["nu_bins"])
    nu_marker     = read(panel["nu_init_marker"])

    # Derive row/col indices from the panel name
    parts = split(pname, "_")
    r_idx = parse(Int, parts[2])
    c_idx = parse(Int, parts[3])

    spec  = row_xspec(Int(nu_init))
    yspec = row_yspec(Int(nu_init))

    ax = Axis(fig[r_idx + 1, c_idx];
              xticks = spec.xticks,
              yticks = yspec.yticks,
              xticklabelsize = tick_fontsize,
              yticklabelsize = tick_fontsize,
              xtickalign = 1.0,
              ytickalign = 1.0,
              xticksmirrored = false,
              yticksmirrored = true,
              limits = (spec.xlims[1], spec.xlims[2], -0.05, yspec.ymax),
              spinewidth = panel_spinewidth,
              xtickwidth = panel_spinewidth,
              ytickwidth = panel_spinewidth,
              xticksize = tick_size, yticksize = tick_size,
              yminorticksize = subtick_size,
              xgridvisible = false, ygridvisible = false,
              xminorticksvisible = false, yminorticksvisible = true,
              yminortickalign = 1.0)

    # Dashed vertical line at initial vibrational state
    vlines!(ax, [nu_marker]; color = :black, linestyle = :dash, linewidth = 1.8)

    # Memory CPA (endpoint kernel)
    mem = panel["memory"]
    mem_prob = read(mem["prob"])
    mem_err  = read(mem["err"])
    if !isempty(mem_prob)
        errorbars!(ax, nu_bins, mem_prob, mem_err;
                   color = color_endpt, whiskerwidth = 12, linewidth = 2.0)
        scatterlines!(ax, nu_bins, mem_prob;
                      color = color_endpt,
                      linewidth = 1.6,
                      markersize = 9,
                      marker = :diamond,
                      strokecolor = :black,
                      strokewidth = 1.2)
    end

    # Markovian CPA
    mark = panel["markovian"]
    mark_prob = read(mark["prob"])
    mark_err  = read(mark["err"])
    if !isempty(mark_prob)
        errorbars!(ax, nu_bins, mark_prob, mark_err;
                   color = color_mark, whiskerwidth = 12, linewidth = 2.0)
        scatterlines!(ax, nu_bins, mark_prob;
                      color = color_mark,
                      linewidth = 1.6,
                      markersize = 9,
                      marker = :circle,
                      strokecolor = :black,
                      strokewidth = 1.2)
    end

    # MD baseline (if present)
    if haskey(panel, "md")
        md = panel["md"]
        md_prob = read(md["prob"])
        md_err  = read(md["err"])
        if !isempty(md_prob)
            errorbars!(ax, nu_bins, md_prob, md_err;
                       color = color_md, whiskerwidth = 12, linewidth = 2.0)
            scatterlines!(ax, nu_bins, md_prob;
                          color = color_md,
                          linewidth = 1.2,
                          markersize = 9,
                          marker = :circle,
                          strokecolor = :black,
                          strokewidth = 1.2)
        end
    end

    # Panel letter annotation (top-left, faux-bold via stroke)
    text!(ax, 0.04, 0.95;
          text        = panel_letter,
          space       = :relative,
          align       = (:left, :top),
          fontsize    = panel_label_fontsize,
          color       = :black,
          strokewidth = 0.5,
          strokecolor = :black)

    axes[r_idx, c_idx] = ax
end

# Link y-axes across each row and hide y-decorations on non-first columns
for r_idx in 1:n_rows
    linkyaxes!(axes[r_idx, :]...)
end
for r_idx in 1:n_rows, c_idx in 2:n_cols
    hideydecorations!(axes[r_idx, c_idx]; grid = false, ticks = false, minorticks = false)
end

# Axis labels
Label(fig[:, 0], "Probability";
      fontsize = label_fontsize,
      rotation = π / 2,
      tellheight = false,
      padding = (0, 0, 0, 0))
Label(fig[n_rows + 2, 1:n_cols], "Final vibrational state ν";
      fontsize = label_fontsize,
      tellwidth = false,
      padding = (0, 0, 0, 0))

# Legend
legend_elems  = Any[
    [LineElement(;   color = color_endpt, linewidth = 1.6),
     MarkerElement(; color = color_endpt, marker = :diamond, markersize = 10,
                     strokecolor = :black, strokewidth = 1.2)],
    [LineElement(;   color = color_mark,  linewidth = 1.6),
     MarkerElement(; color = color_mark,  marker = :circle, markersize = 10,
                     strokecolor = :black, strokewidth = 1.2)],
    LineElement(; color = :black, linestyle = :dash, linewidth = 1.8),
]
legend_labels = String[
    "Memory",
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
       nbanks = 1,
       tellwidth = false,
       tellheight = true,
       framevisible = false,
       framecolor = :black,
       framewidth = panel_spinewidth,
       labelsize = legend_fontsize,
       patchsize = (24, 12),
       margin = (0, 0, 0, 0))

# Row and column gaps (identical to original)
rowgap!(fig.layout, 1, 2)
for r_idx in 2:n_rows
    rowgap!(fig.layout, r_idx, 4)
end
rowgap!(fig.layout, n_rows + 1, 0)
rowgap!(fig.layout, n_rows + 2, 4)
colgap!(fig.layout, 0)

close(h5)

display(fig)

outdir = plotsdir("fig_5")
mkpath(outdir)
save(joinpath(outdir, "fig_5.pdf"), fig)
