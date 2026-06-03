## Only for the main process
using Distributed
using DrWatson
@quickactivate "HokseonReproduce" ## Activate project everywhere
import Pkg; Pkg.precompile() ## Precompile packages in master to speed up workers' precompilation
using HDF5
using DelimitedFiles
using HokseonAssistant
using CairoMakie
using LinearAlgebra: eigmin, eigmax, tr
using HokseonPlots
using ColorSchemes
using NQCModels
using NQCDynamics
using NQCCalculators
using Colors
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;
HokseonAssistant.julia_build_procs()


# Load packages everywhere
@everywhere using HokseonReproduce
@everywhere using Unitful, UnitfulAtomic
@everywhere using StaticArrays: SA


function buildMarkovianFriction(params_dict::Dict{String, Any})
    @unpack r, z, temperature = params_dict

    pogo           = POGOModel()
    M              = 300
    bw             = austrip(900u"eV")
    model          = WideBandBath(pogo; step=(2bw)/M, bandmin=-bw, bandmax=bw)
    #μ_NO           = austrip((14.007 * 15.999 / (14.007 + 15.999)) * u"u")
    μ_NO           = austrip(1000u"u")   # reduced mass of NO, as approximation for both DOF
    atoms          = Atoms([μ_NO])
    temperature_au = austrip(temperature * u"K")

    sim = Simulation{DiabaticMDEF}(atoms, model,
              friction_method=NQCCalculators.WideBandExact(model.ρ, 1/temperature_au),
              temperature=temperature_au)

    r_au = austrip(r * u"Å")
    z_au = austrip(z * u"Å")
    return NQCCalculators.evaluate_friction(sim.cache, hcat([r_au, z_au]))   # full 2×2 γ matrix
end

function buildElementFrequencyLambda(params_dict::Dict{String, Any})
    @unpack r, z, temperature, energy = params_dict

    adsorbate_m      = NOAuAdsorbate()
    energy_au        = austrip.(energy * u"eV")
    temperature_au   = austrip.(temperature * u"K")
    configuration_au = SA[austrip(r * u"Å"), austrip(z * u"Å")]  # SA[r, z]

    # Vector (one per ω) of Symmetric 2×2 friction matrices Λ(ω,x).
    return FrequencyLambda.Lambda(energy_au, adsorbate_m, configuration_au, temperature_au)
end

# Panels: 3×3 grid. Each ROW fixes a bond length r (Å); the three COLUMNS show
# the two diagonal elements and the off-diagonal element of the friction matrix
# Λ(ω,x). Within each panel, for every molecule–surface distance z (one colour
# each) we draw the frequency-dependent Λ element as a SOLID curve and its
# Markovian counterpart as a DASHED horizontal line.
const R_VALUES    = [1.17, 1.6, 2.0]                  # one panel ROW per r / Å
const Z_VALUES    = [1.6, 2, 3]                    # molecule–surface distance / Å (one curve each)
const TEMPERATURE = 300                            # single electronic temperature / K
const ENERGY_GRID = collect(0.01:0.01:20.0)        # ω / eV  (avoid 0 — Lambda divides by ω)

# Matrix DOF: 1 ≡ r (N–O bond length), 2 ≡ z (molecule–surface distance).
# Columns of the figure: Λ₁₁ (r,r), Λ₂₂ (z,z), Λ₁₂ (r,z).
const ELEMENTS       = [(1, 1), (2, 2), (1, 2)]    # one panel COLUMN per element
const ELEMENT_TITLES = ["(r,r)", "(z,z)", "(r,z)"]

# ─────────────────────────────────────────────────────────────────
# Figure: 3×3 panels for a CENTRED two-column (figure*) placement. In revtex,
# inside `figure*` the `\linewidth` is the full two-column text width, so
#     \includegraphics[width=0.7\linewidth]{fig/.../LambdaElement_r.pdf}
# renders the PDF at 0.7 of that width. We author the PDF at exactly that final
# size (FIG_FRACTION · TWO_COLUMN_WIDTH) so it embeds at scale 1.0 — 1 Makie pt
# = 1 typeset pt — keeping fonts and line widths crisp and exactly as set here.
# Style follows friction/NOAu/plot_LambdaAveraged_r.jl; every length below is a
# full-size design value times FIG_FRACTION (so proportions are preserved).
# ─────────────────────────────────────────────────────────────────
const MINION_FONT  = projectdir("fonts", "MinionPro-Capt.otf")
const FIG_FRACTION = 0.7                                 # matches width=0.7\linewidth
scaled(x) = x .* FIG_FRACTION                            # works on scalars and tuples

const FIG_WIDTH  = round(Int, FIG_FRACTION * HokseonPlots.TWO_COLUMN_WIDTH)
const FIG_HEIGHT = round(Int, FIG_WIDTH * 0.9)

const panel_spinewidth = scaled(1.2)
const LINE_WIDTH       = scaled(1.3)   # data curves, Markovian dashes, legend swatches
const PANEL_COLGAP     = scaled(10)    # gap between element columns / row-label column
const FIG_COLGAP       = scaled(4)     # gap between y-label and panel block

# Font hierarchy (full-size base × FIG_FRACTION) for the compact 3×3 layout.
const AXIS_LABEL_SIZE   = scaled(16)   # shared x / y labels (single, page-spanning)
const TITLE_SIZE        = scaled(15)   # per-column element headers / row labels
const TICK_LABEL_SIZE   = scaled(10)
const ANNOTATION_SIZE   = scaled(10)
const LEGEND_LABEL_SIZE = scaled(11)
const LEGEND_TITLE_SIZE = scaled(12)

colors    = [colorant"#1f77b4", colorant"#2ca02c", colorant"#d62728"]   # one per z
unit_conv = ustrip(auconvert(u"u", 1) / auconvert(u"ps", 1))            # au friction → u⋅ps⁻¹

const N_ROWS = length(R_VALUES)
const N_COLS = length(ELEMENTS)

fig = Figure(size=(FIG_WIDTH, FIG_HEIGHT),
             figure_padding=(2, 6, 2, 2),
             fonts=(;regular=MINION_FONT))

# Panels live in a nested grid: rows glued (rowgap = 0, ω-axis shared so only
# the bottom row needs x-tick labels); columns kept apart (each element has its
# own y-scale and therefore its own y-tick labels).
panel_grid = GridLayout(fig[1, 2])
axes = Matrix{Any}(undef, N_ROWS, N_COLS)

for row in 1:N_ROWS, col in 1:N_COLS
    is_bottom = (row == N_ROWS)
    is_top    = (row == 1)
    axes[row, col] = MyAxis(panel_grid[row, col];
                title              = is_top ? ELEMENT_TITLES[col] : "",
                titlesize          = TITLE_SIZE,
                titlefont          = MINION_FONT,
                xlabel             = "",
                ylabel             = "",
                xticklabelsize     = TICK_LABEL_SIZE,
                yticklabelsize     = TICK_LABEL_SIZE,
                xticklabelsvisible = is_bottom,   # only the bottom row labels x
                limits             = (0.0, nothing, nothing, nothing),
                xgridvisible       = false, ygridvisible = false,
                xticksmirrored     = false,
                yticksmirrored     = true,   # mirror y-ticks onto right spine
                xtickalign         = 1,      # ticks point INTO the axis
                ytickalign         = 1,
                xminortickalign    = 1,
                yminortickalign    = 1,
                spinewidth         = panel_spinewidth,
                xtickwidth         = panel_spinewidth,
                ytickwidth         = panel_spinewidth)
end

linkxaxes!(vec(axes)...)        # shared ω-axis across every panel
rowgap!(panel_grid, 0)          # glue the rows
colgap!(panel_grid, PANEL_COLGAP)   # keep the element columns apart

for (row, r) in enumerate(R_VALUES)
    # Only z varies within a row → deterministic z → colour mapping.
    zs = sort(Z_VALUES)

    for (i, z) in enumerate(zs)
        params_dict = Dict{String, Any}(
            "r"           => r,
            "z"           => z,
            "temperature" => TEMPERATURE,
            "energy"      => ENERGY_GRID,
        )

        # Compute the full matrix ONCE per (r, z), then fan the elements out
        # across the three columns.
        Λ_mats = buildElementFrequencyLambda(params_dict)   # Vector of 2×2 Symmetric
        γ_mat  = buildMarkovianFriction(params_dict)         # 2×2 matrix

        ω_eV = ENERGY_GRID   # grid is already in eV

        for (col, (k, l)) in enumerate(ELEMENTS)
            # frequency-dependent element Λₖₗ(ω) — solid curve
            Λ_kl = getindex.(Λ_mats, k, l) .* unit_conv
            # Markovian element γₖₗ (scalar) — dashed horizontal line
            γ_kl = γ_mat[k, l] * unit_conv

            lines!(axes[row, col],  ω_eV, Λ_kl; color=colors[i], linestyle=:solid, linewidth=LINE_WIDTH)
            hlines!(axes[row, col], [γ_kl];     color=colors[i], linestyle=:dash,  linewidth=LINE_WIDTH)
        end
    end

    # Per-row fixed-r label on the RIGHT margin, acting as a rotated row header.
    # Placed in an extra panel_grid column so it stays aligned with the row.
    Label(panel_grid[row, N_COLS + 1], "r = $r Å";
          rotation = -π/2, tellheight = false, tellwidth = true,
          fontsize = TITLE_SIZE, padding = scaled((4, 0, 0, 0)))
end

# The row-label column was added inside the loop, so re-assert a uniform column
# gap to cover the new panel↔label gap as well.
colgap!(panel_grid, PANEL_COLGAP)

# Global system / temperature label in the top-left panel.
#text!(axes[1, 1], 0.97, 0.94;
#      text  = "NOAuAdsorbate\nT = $TEMPERATURE K",
#      space = :relative, align = (:right, :top), fontsize = ANNOTATION_SIZE)

# Shared rotated y-label spanning all rows; shared x-label spanning all columns.
Label(fig[1, 1], "Kμν(ω,x)  (u⋅ps⁻¹)";
      rotation = π/2, tellheight = false, tellwidth = true,
      fontsize = AXIS_LABEL_SIZE, padding = (0, 0, 0, 0))
Label(fig[2, 2], "ħω  (eV)";
      tellwidth = false, tellheight = true,
      fontsize = AXIS_LABEL_SIZE, padding = (0, 0, 2, 0))

# Two-group horizontal legend across the top:
#   colour group → z (Å);  style group → Frequency (solid) vs Markovian (dash)
color_elems  = [LineElement(color=colors[i], linewidth=LINE_WIDTH) for i in 1:length(Z_VALUES)]
color_labels = ["$z" for z in sort(Z_VALUES)]

style_elems  = [LineElement(color=:black, linestyle=:solid, linewidth=LINE_WIDTH),
                LineElement(color=:black, linestyle=:dash,  linewidth=LINE_WIDTH)]
style_labels = ["Frequency", "Markovian"]

legend_kwargs = (
    orientation   = :horizontal,
    titleposition = :left,
    nbanks        = 1,
    tellwidth     = false,
    tellheight    = true,
    halign        = :center,
    framevisible  = false,
    patchsize     = scaled((26, 12)),
    colgap        = scaled(8),
    groupgap      = scaled(24),
    titlegap      = scaled(8),
    patchlabelgap = scaled(4),
    padding       = scaled((2, 2, 1, 1)),
    labelsize     = LEGEND_LABEL_SIZE,
    titlesize     = LEGEND_TITLE_SIZE,
    labelfont     = MINION_FONT,
    titlefont     = MINION_FONT,
)

# Single multi-group legend spanning BOTH figure columns (y-label + panels) so
# `halign = :center` centres it on the full figure width, not just the panels.
Legend(fig[0, 1:2],
       [color_elems, style_elems],
       [color_labels, style_labels],
       ["z  (Å)", "Friction"];
       legend_kwargs...)

colgap!(fig.layout, FIG_COLGAP)

display(fig)

outpath = plotsdir("friction", "NOAu", "LambdaElement_r.pdf")
mkpath(dirname(outpath))
save(outpath, fig)
