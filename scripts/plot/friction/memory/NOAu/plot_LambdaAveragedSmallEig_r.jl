## Only for the main process
using Distributed
using DrWatson
@quickactivate "HokseonReproduce" ## Activate project everywhere
import Pkg; Pkg.precompile() ## Precompile packages in master to speed up workers' precompilation
using HDF5
using DelimitedFiles
using HokseonAssistant
using CairoMakie
using LinearAlgebra: eigmin, eigmax, tr, Symmetric
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

function buildFrequencyLambdaMatrices(params_dict::Dict{String, Any})
    @unpack r, z, temperature, energy = params_dict

    adsorbate_m      = NOAuAdsorbate()
    energy_au        = austrip.(energy * u"eV")
    temperature_au   = austrip.(temperature * u"K")
    configuration_au = SA[austrip(r * u"Å"), austrip(z * u"Å")]  # SA[r, z]

    # Vector (one per ω) of 2×2 friction matrices Λ(ω,x). We reduce each matrix
    # two ways (trace-average and smallest eigenvalue) for the two columns.
    return FrequencyLambda.Lambda(energy_au, adsorbate_m, configuration_au, temperature_au)
end

# Panels: 3×2 grid. Each ROW fixes a bond length r (Å). The LEFT column shows
# the trace-averaged friction ⟨K⟩ = tr(Λ)/ndof; the RIGHT column shows the
# smallest eigenvalue λ_min(Λ) (friction along the least-damped direction).
# Both columns come from the SAME FrequencyLambda.Lambda(ω) call per (r,z) —
# we just reduce the resulting matrices differently. Within each panel, for
# every molecule–surface distance z (one colour) we draw the frequency-dependent
# reduction as a SOLID curve and its Markovian counterpart as a DASHED line.
const R_VALUES    = [1.17, 1.6, 2.0]                # one panel ROW per r / Å
const Z_VALUES    = [1.6, 2, 3]                    # molecule–surface distance / Å (one curve each)
const TEMPERATURE = 300                            # single electronic temperature / K
# ω grid (eV); avoid 0 — Lambda divides by ω. The left column (⟨K⟩) spans
# 0–20 eV; the right column (λ_min) is extended to 0–40 eV. Because 0.01:0.01:20
# is the exact prefix of 0.01:0.01:40 (same start/step), ONE Lambda call over
# the 0–40 grid serves BOTH columns — the left simply reuses the 0–20 prefix.
const ENERGY_LEFT   = collect(0.01:0.01:20.0)
const ENERGY_RIGHT  = collect(0.01:0.01:40.0)

# Two columns ≡ two scalar reductions of the 2×2 friction matrix, each with its
# own ω sub-range (left = 0–20 eV, right = 0–40 eV).
const COLUMN_TITLES   = ["⟨K(ω,x)⟩", "λₘᵢₙ(ω,x)"]
const COLUMN_ENERGY   = [ENERGY_LEFT, ENERGY_RIGHT]
const COLUMN_REDUCERS = [
    M -> tr(M) / size(M, 1),     # left  : trace-averaged friction
    M -> eigmin(Symmetric(M)),   # right : smallest eigenvalue (least-damped direction)
]

# ─────────────────────────────────────────────────────────────────
# Figure: trace-averaged friction (left) and smallest eigenvalue (right) vs ω,
# with three rows of bond length r glued together. Setup/style follows
# friction/NOAu/plot_LambdaAveraged_r.jl (PRX font hierarchy, thick spines).
# A single shared "Friction" y-label sits on the far left; the per-row r labels
# sit on the right margin; column headers name the two reductions.
# ─────────────────────────────────────────────────────────────────
const MINION_FONT      = projectdir("fonts", "MinionPro-Capt.otf")
const panel_spinewidth = 1.8

# PRX-standard font hierarchy (see plot_LambdaAveraged_r.jl)
const AXIS_LABEL_SIZE   = 24
const TITLE_SIZE        = 22
const TICK_LABEL_SIZE   = 20
const ANNOTATION_SIZE   = 20
const LEGEND_LABEL_SIZE = 16
const LEGEND_TITLE_SIZE = 18

colors    = [colorant"#1f77b4", colorant"#2ca02c", colorant"#d62728"]   # one per z
unit_conv = ustrip(auconvert(u"u", 1) / auconvert(u"ps", 1))            # au friction → u⋅ps⁻¹

const N_ROWS = length(R_VALUES)
const N_COLS = length(COLUMN_TITLES)

fig = Figure(size=(HokseonPlots.TWO_COLUMN_WIDTH, HokseonPlots.TWO_COLUMN_WIDTH * 1.25),
             figure_padding=(2, 6, 2, 2),
             fonts=(;regular=MINION_FONT))

# Panels live in a nested grid: rows glued (rowgap = 0, ω-axis shared so only
# the bottom row needs x-tick labels); columns kept apart (each reduction has
# its own y-scale and therefore its own y-tick labels).
panel_grid = GridLayout(fig[1, 2])
axes = Matrix{Any}(undef, N_ROWS, N_COLS)

for row in 1:N_ROWS, col in 1:N_COLS
    is_bottom = (row == N_ROWS)
    is_top    = (row == 1)
    axes[row, col] = MyAxis(panel_grid[row, col];
                title              = is_top ? COLUMN_TITLES[col] : "",
                titlesize          = TITLE_SIZE,
                titlefont          = MINION_FONT,
                xlabel             = "",
                ylabel             = "",
                xticklabelsize     = TICK_LABEL_SIZE,
                yticklabelsize     = TICK_LABEL_SIZE,
                xticklabelsvisible = is_bottom,   # only the bottom row labels x
                limits             = (0.0, maximum(COLUMN_ENERGY[col]), nothing, nothing),
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

for col in 1:N_COLS
    linkxaxes!(axes[:, col]...)   # share ω-axis WITHIN a column (ranges differ between columns)
end
rowgap!(panel_grid, 0)     # glue the rows
colgap!(panel_grid, 14)    # keep the two reduction columns apart

for (row, r) in enumerate(R_VALUES)
    # Only z varies within a row → deterministic z → colour mapping.
    zs = sort(Z_VALUES)

    for (i, z) in enumerate(zs)
        params_dict = Dict{String, Any}(
            "r"           => r,
            "z"           => z,
            "temperature" => TEMPERATURE,
            "energy"      => ENERGY_RIGHT,   # compute over the wider 0–40 eV grid
        )

        # ONE Lambda(ω) call per (r, z), over the FULL 0–40 eV grid. Each column
        # then uses its own ω sub-range (left = 0–20 prefix, right = full 0–40).
        Λ_mats = buildFrequencyLambdaMatrices(params_dict)   # 2×2 matrices over ENERGY_RIGHT
        γ_mat  = buildMarkovianFriction(params_dict)          # 2×2 matrix

        for (col, reduce_fn) in enumerate(COLUMN_REDUCERS)
            ω_eV  = COLUMN_ENERGY[col]               # this column's ω grid (eV)
            Λ_col = @view Λ_mats[1:length(ω_eV)]     # 0–20 prefix (left) or all 0–40 (right)

            # frequency-dependent reduction — solid curve
            curve = reduce_fn.(Λ_col) .* unit_conv
            # Markovian reduction (scalar) — dashed horizontal line
            mark  = reduce_fn(γ_mat) * unit_conv

            lines!(axes[row, col],  ω_eV, curve; color=colors[i], linestyle=:solid, linewidth=2)
            hlines!(axes[row, col], [mark];      color=colors[i], linestyle=:dash,  linewidth=2)
        end
    end

    # Per-row fixed-r label on the RIGHT margin, acting as a rotated row header.
    # Placed in an extra panel_grid column so it stays aligned with the row.
    Label(panel_grid[row, N_COLS + 1], "r = $r Å";
          rotation = -π/2, tellheight = false, tellwidth = true,
          fontsize = ANNOTATION_SIZE, padding = (6, 0, 0, 0))
end

# The row-label column was added inside the loop, so re-assert a uniform column
# gap to cover the new panel↔label gap as well.
colgap!(panel_grid, 14)

# Single shared rotated y-label on the far left (both columns are frictions).
Label(fig[1, 1], "Friction  (u⋅ps⁻¹)";
      rotation = π/2, tellheight = false, tellwidth = true,
      fontsize = AXIS_LABEL_SIZE, padding = (0, 0, 0, 0))
# Shared x-label spanning both columns at the bottom.
Label(fig[2, 2], "ħω  (eV)";
      tellwidth = false, tellheight = true,
      fontsize = AXIS_LABEL_SIZE, padding = (0, 0, 2, 0))

# Two-group horizontal legend across the top:
#   colour group → z (Å);  style group → Frequency (solid) vs Markovian (dash)
color_elems  = [LineElement(color=colors[i], linewidth=2) for i in 1:length(Z_VALUES)]
color_labels = ["$z" for z in sort(Z_VALUES)]

style_elems  = [LineElement(color=:black, linestyle=:solid, linewidth=2),
                LineElement(color=:black, linestyle=:dash,  linewidth=2)]
style_labels = ["Frequency", "Markovian"]

# Span BOTH figure columns (y-label + panels) so the centred halign references
# the full figure width, not just the panel block.
legend_grid = GridLayout(fig[0, 1:2])

legend_kwargs = (
    orientation   = :horizontal,
    titleposition = :left,
    nbanks        = 1,
    tellwidth     = false,
    tellheight    = true,
    halign        = :center,
    framevisible  = false,
    patchsize     = (26, 12),
    colgap        = 8,
    titlegap      = 8,
    patchlabelgap = 4,
    padding       = (2, 2, 1, 1),
    labelsize     = LEGEND_LABEL_SIZE,
    titlesize     = LEGEND_TITLE_SIZE,
    labelfont     = MINION_FONT,
    titlefont     = MINION_FONT,
)

Legend(legend_grid[1, 1], color_elems, color_labels, "z  (Å)";   legend_kwargs...)
Legend(legend_grid[2, 1], style_elems, style_labels, "Friction"; legend_kwargs...)
rowgap!(legend_grid, 2)

colgap!(fig.layout, 4)

display(fig)

outpath = plotsdir("friction", "NOAu", "LambdaAveragedSmallEig.pdf")
mkpath(dirname(outpath))
save(outpath, fig)
