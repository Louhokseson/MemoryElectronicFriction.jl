## Only for the main process
using Distributed
using DrWatson
@quickactivate "HokseonReproduce" ## Activate project everywhere
import Pkg; Pkg.precompile() ## Precompile packages in master to speed up workers' precompilation
using HDF5
using DelimitedFiles
using HokseonAssistant
using CairoMakie
using LinearAlgebra: eigmin, eigmax, tr, Symmetric, eigen
using HokseonPlots
using ColorSchemes
using NQCModels
using NQCDynamics
using NQCCalculators
using StaticArrays: SMatrix, SVector
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

function threshold_energy(params_dict::Dict{String, Any})
    @unpack r, z, temperature, energy = params_dict
    Λ_mats = buildElementFrequencyLambda(params_dict)   # Vector of 2×2 Symmetric

    @inline function min_eigvec(M)
    F = eigen(Symmetric(SMatrix{2,2,Float64}(M)))  # analytic 2×2; values ascending
    return F.vectors[:, 1]                          # SVector{2} → eigvec of λ_min
    end

    vmin = min_eigvec(Λ_mats[1])

    adsorbate_m      = NOAuAdsorbate()
    temperature_au = austrip(temperature * u"K")

    ∇h = vmin' * HokseonReproduce.dh_dx(r, z, adsorbate_m)   # SVector{2} → scalar
    ∇Δ = vmin' * HokseonReproduce.dΔ_dx(r, z, adsorbate_m)   # SVector{2} → scalar
    Δ = HokseonReproduce.Δ(z, adsorbate_m)   # scalar
    h = HokseonReproduce.adsorbate_h(r, z, adsorbate_m)   # scalar

    threshold_energy = sqrt(sqrt(6)*abs(∇h / ∇Δ * Δ - h)^2 + (2π^2 * temperature_au^2))

    return threshold_energy
end

# Panels: 2×4 grid. Each ROW fixes a bond length r (Å). The first three COLUMNS
# show the two diagonal elements and the off-diagonal element of the friction
# matrix Λ(ω,x); the fourth COLUMN shows its smallest eigenvalue λ_min (friction
# along the least-damped direction). Within each panel, for every molecule–
# surface distance z (one colour each) we draw the frequency-dependent quantity
# as a SOLID curve and its Markovian counterpart as a DASHED horizontal line.
const R_VALUES    = [1.17, 1.6]                  # one panel ROW per r / Å
const Z_VALUES    = [1.7, 2, 3]                    # molecule–surface distance / Å (one curve each)
const TEMPERATURE = 300                            # single electronic temperature / K
const ENERGY_GRID = collect(0.01:0.01:20.0)        # ω / eV  (avoid 0 — Lambda divides by ω)

# Matrix DOF: 1 ≡ r (N–O bond length), 2 ≡ z (molecule–surface distance). Each
# column reduces the 2×2 friction matrix Λ(ω,x) to one scalar: the first three
# pick a matrix element, the fourth takes the smallest eigenvalue.
const COLUMN_TITLES   = ["(r,r)", "(z,z)", "(r,z)", "λₘᵢₙ  (×10⁻³)"]
const COLUMN_REDUCERS = [
    M -> M[1, 1],                # Λ₁₁ (r,r)
    M -> M[2, 2],                # Λ₂₂ (z,z)
    M -> M[1, 2],                # Λ₁₂ (r,z)
    M -> eigmin(Symmetric(M)),   # smallest eigenvalue (least-damped direction)
]
# λ_min is ~10⁻³ of the matrix elements; scale that column up by 10³ so its
# y-ticks read O(1) (the header carries the ×10⁻³). 1.0 ≡ no display rescaling.
const COLUMN_SCALE    = [1.0, 1.0, 1.0, 1e3]

# ─────────────────────────────────────────────────────────────────
# Figure: 2×4 panels for a CENTRED two-column (figure*) placement. In revtex,
# inside `figure*` the `\linewidth` is the full two-column text width, so
#     \includegraphics[width=\linewidth]{fig/.../LambdaElement_r.pdf}
# embeds the PDF at the full two-column text width. We author at exactly that
# size (FIG_FRACTION = 1.0 · TWO_COLUMN_WIDTH) so it embeds at scale 1.0 — 1
# Makie pt = 1 typeset pt — keeping fonts and line widths exactly as set here.
# The figure is deliberately LANDSCAPE (height ≈ 0.45·width) so the eight panels
# read WIDE rather than tall. Style follows friction/NOAu/plot_LambdaAveraged_r.jl.
# ─────────────────────────────────────────────────────────────────
const MINION_FONT  = projectdir("fonts", "MinionPro-Capt.otf")
const FIG_FRACTION = 1.2                                 # author at the full two-column width (figure*), embed at scale 1.0
scaled(x) = x .* FIG_FRACTION                            # works on scalars and tuples

const FIG_WIDTH  = round(Int, FIG_FRACTION * HokseonPlots.TWO_COLUMN_WIDTH)
const FIG_HEIGHT = round(Int, FIG_WIDTH * 0.55)          # LANDSCAPE: wide, short panels (was 0.9 → tall & thin)

const panel_spinewidth = scaled(1.2)
const LINE_WIDTH       = scaled(1.3)   # data curves, Markovian dashes, legend swatches
const MARKER_SIZE      = scaled(7)     # threshold-energy × markers (plot + legend swatch)
const PANEL_COLGAP     = scaled(10)    # gap between element columns / row-label column
const FIG_COLGAP       = scaled(4)     # gap between y-label and panel block

# Font hierarchy (full-size base × FIG_FRACTION) for the compact 2×4 layout.
const AXIS_LABEL_SIZE   = scaled(13)   # shared x / y labels (single, page-spanning)
const TITLE_SIZE        = scaled(12)   # per-column element headers / row labels
const TICK_LABEL_SIZE   = scaled(10)
const ANNOTATION_SIZE   = scaled(10)
const LEGEND_LABEL_SIZE = scaled(10)    # compact, PRX-scale legend (fits 3 groups in one row)
const LEGEND_TITLE_SIZE = scaled(10.5)

colors    = [colorant"#1f77b4", colorant"#2ca02c", colorant"#d62728"]   # one per z
unit_conv = ustrip(auconvert(u"u", 1) / auconvert(u"ps", 1))            # au friction → u⋅ps⁻¹

const N_ROWS = length(R_VALUES)
const N_COLS = length(COLUMN_TITLES)

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
    is_lmin   = (col == N_COLS)   # λ_min column → log-ω so threshold markers spread out
    axes[row, col] = MyAxis(panel_grid[row, col];
                title              = is_top ? COLUMN_TITLES[col] : "",
                titlesize          = TITLE_SIZE,
                titlefont          = MINION_FONT,
                xlabel             = "",
                ylabel             = "",
                xticklabelsize     = TICK_LABEL_SIZE,
                yticklabelsize     = TICK_LABEL_SIZE,
                xticklabelsvisible = is_bottom,   # only the bottom row labels x
                xscale             = is_lmin ? log10 : identity,   # λ_min col on log-ω
                limits             = is_lmin ? (0.008, nothing, nothing, nothing) :
                                               (0.0,   nothing, nothing, nothing),
                yautolimitmargin   = (0.05, 0.28),   # top headroom: clears Markovian dashes AND leaves room for the (a)-(h) panel label
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

# Element columns share one LINEAR ω-axis; the λ_min column keeps its own LOG
# ω-axis (linked only across its two rows) so its threshold markers spread out.
linkxaxes!(vec(axes[:, 1:N_COLS-1])...)
linkxaxes!(axes[:, N_COLS]...)
rowgap!(panel_grid, scaled(12))     # small gap so the two rows' boundary y-ticks (e.g. 0 vs 30) don't collide
colgap!(panel_grid, PANEL_COLGAP)   # keep the element columns apart

# λ_min-column threshold-energy markers, collected here and drawn in a SECOND
# pass so each × sits on top of all the λ_min curves in its panel.
threshold_markers = Tuple{Int, Float64, eltype(colors)}[]

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

        # λ_min-column threshold energy ħω* — one × marker per (r, z), col N_COLS.
        threshold_energy_eV = austrip(threshold_energy(params_dict) * u"eV")
        push!(threshold_markers, (row, threshold_energy_eV, colors[i]))

        for (col, reduce_fn) in enumerate(COLUMN_REDUCERS)
            scale = unit_conv * COLUMN_SCALE[col]   # au→u⋅ps⁻¹, plus this column's display rescale
            # frequency-dependent reduction of Λ(ω) — solid curve
            curve = reduce_fn.(Λ_mats) .* scale
            # Markovian reduction of γ (scalar) — dashed horizontal line
            mark  = reduce_fn(γ_mat) * scale

            lines!(axes[row, col],  ω_eV, curve; color=colors[i], linestyle=:solid, linewidth=LINE_WIDTH)
            hlines!(axes[row, col], [mark];      color=colors[i], linestyle=:dash,  linewidth=LINE_WIDTH)
        end
    end

    # Per-row fixed-r label on the RIGHT margin, acting as a rotated row header.
    # Placed in an extra panel_grid column so it stays aligned with the row.
    Label(panel_grid[row, N_COLS + 1], "r = $r Å";
          rotation = π/2, tellheight = false, tellwidth = true,
          fontsize = TITLE_SIZE, padding = scaled((4, 0, 0, 0)))
end

# The row-label column was added inside the loop, so re-assert a uniform column
# gap to cover the new panel↔label gap as well.
colgap!(panel_grid, PANEL_COLGAP)

# Second pass: λ_min-column threshold-energy × markers, drawn on top of every
# curve (col N_COLS only; their x sits on that column's log-ω axis).
for (row, x, c) in threshold_markers
    scatter!(axes[row, N_COLS], [x], [0]; color=c, marker=:x, markersize=MARKER_SIZE)
end

# Panel labels (a)-(h), row-major (top row a-d, bottom row e-h), placed in the
# top-left headroom opened up by the widened yautolimitmargin. Drawn last so they
# sit on top of every curve.
for row in 1:N_ROWS, col in 1:N_COLS
    letter = 'a' + (row - 1) * N_COLS + (col - 1)
    # Bold (PRX convention) via faux-bold stroke — no bold MinionPro file ships
    # with the project, so stroking the glyphs in their own colour keeps the Minion
    # typeface instead of falling back to a generic :bold sans-serif.
    text!(axes[row, col], 0.04, 0.95; text="($(letter))", space=:relative,
          align=(:left, :top), fontsize=ANNOTATION_SIZE,
          strokewidth=scaled(0.4), strokecolor=:black)
end

# Global system / temperature label in the top-left panel.
#text!(axes[1, 1], 0.97, 0.94;
#      text  = "NOAuAdsorbate\nT = $TEMPERATURE K",
#      space = :relative, align = (:right, :top), fontsize = ANNOTATION_SIZE)

# Shared rotated y-label spanning all rows; shared x-label spanning all columns.
Label(fig[1, 1], "Friction  (u⋅ps⁻¹)";
      rotation = π/2, tellheight = false, tellwidth = true,
      fontsize = AXIS_LABEL_SIZE, padding = (0, 0, 0, 0))
Label(fig[2, 2], "ħω  (eV)";
      tellwidth = false, tellheight = true,
      fontsize = AXIS_LABEL_SIZE, padding = (0, 0, 2, 0))

# Three-group horizontal legend across the top:
#   colour group → z (Å);  style group → Frequency (solid) vs Markovian (dash);
#   threshold group → × marker = threshold frequency ħω*
color_elems  = [LineElement(color=colors[i], linewidth=LINE_WIDTH) for i in 1:length(Z_VALUES)]
color_labels = ["$z" for z in sort(Z_VALUES)]

style_elems  = [LineElement(color=:black, linestyle=:solid, linewidth=LINE_WIDTH),
                LineElement(color=:black, linestyle=:dash,  linewidth=LINE_WIDTH)]
style_labels = ["Memory", "Markovian"]

# Neutral black × (colour already encodes z) matching the threshold scatter marker.
threshold_elems  = [MarkerElement(marker=:x, color=:black, markersize=MARKER_SIZE)]
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
    groupgap      = scaled(0),    # group separation comes ENTIRELY from the spaces around the "|" titles below; see note at the Legend() call
    titlegap      = scaled(4),
    patchlabelgap = scaled(3),
    padding       = scaled((2, 2, 1, 1)),
    labelsize     = LEGEND_LABEL_SIZE,
    titlesize     = LEGEND_TITLE_SIZE,
    labelfont     = MINION_FONT,
    titlefont     = MINION_FONT,
)

# Single multi-group legend spanning BOTH figure columns (y-label + panels) so
# `halign = :center` centres it on the full figure width, not just the panels.
#
# The "|" group separators are TITLE prefixes (titleposition = :left). To centre
# each "|" in the gap between groups we zero `groupgap` and pad the pipe with an
# EQUAL number of spaces on both sides: with no `groupgap`, the only whitespace
# left of the pipe is the leading spaces here, so leading == trailing ⇒ the pipe
# is centred by construction (verified: ~67 px vs ~68 px either side at 600 dpi),
# independent of font metrics. Keep the two prefixes identical.
Legend(fig[0, 1:2],
       [color_elems, style_elems, threshold_elems],
       [color_labels, style_labels, threshold_labels],
       ["z  (Å)", "   |   Friction", "   |   PSD threshold"];
       legend_kwargs...)

colgap!(fig.layout, FIG_COLGAP)
rowgap!(fig.layout, scaled(2))   # pull the legend row (and x-label) tight to the panels

display(fig)

outpath = plotsdir("friction", "NOAu", "LambdaElement_r.pdf")
mkpath(dirname(outpath))
save(outpath, fig)
