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
    return NQCCalculators.evaluate_friction(sim.cache, hcat([r_au, z_au]))
end

function buildAveragedMarkovianFriction(params_dict::Dict{String, Any})
    γ_mat = buildMarkovianFriction(params_dict)
    return tr(γ_mat) / size(γ_mat, 1)
end

function buildSystemBath(params_dict::Dict{String, Any})
    @unpack r, z, temperature, energy = params_dict

    adsorbate_m      = NOAuAdsorbate()
    energy_au        = austrip.(energy * u"eV")
    temperature_au   = austrip.(temperature * u"K")
    configuration_au = SA[austrip(r * u"Å"), austrip(z * u"Å")]  # SA[r, z]

    return adsorbate_m, configuration_au, energy_au, temperature_au
end

function buildAveragedFrequencyLambda(params_dict::Dict{String, Any})
    @unpack r, z, temperature, energy = params_dict

    adsorbate_m      = NOAuAdsorbate()
    energy_au        = austrip.(energy * u"eV")
    temperature_au   = austrip.(temperature * u"K")
    configuration_au = SA[austrip(r * u"Å"), austrip(z * u"Å")]  # SA[r, z]

    return FrequencyLambda.LambdaAveraged(energy_au, adsorbate_m, configuration_au, temperature_au)
end

# Panels: one per N–O bond length r (Å). Within each panel we draw the same
# set of molecule–surface distances z (one colour each).
const R_VALUES    = [1.17, 1.6]                  # one stacked panel per r / Å
const Z_VALUES    = [1.6, 2, 3]                    # molecule–surface distance / Å (one curve each)
const TEMPERATURE = 300                            # single electronic temperature / K
const ENERGY_GRID = collect(0.01:0.01:20.0)        # ω / eV  (avoid 0 — Lambda divides by ω)

# ─────────────────────────────────────────────────────────────────
# Figure: trace-averaged friction vs ω, with three panels stacked in a
# column and attached to one another. Each panel fixes a bond length r;
# within it, for each molecule–surface distance z we draw the
# frequency-dependent LambdaAveraged as a SOLID curve and the Markovian
# average as a DASHED horizontal line (a scalar), sharing the z colour.
# Style follows friction/ErpenbeckThoss/plot_Lambda_Gamma.jl.
# ─────────────────────────────────────────────────────────────────
const MINION_FONT      = projectdir("fonts", "MinionPro-Capt.otf")
const panel_spinewidth = 1.8

# PRX-standard font hierarchy (see plot_Lambda_Gamma.jl)
const AXIS_LABEL_SIZE   = 24
const TICK_LABEL_SIZE   = 20
const ANNOTATION_SIZE   = 20
const LEGEND_LABEL_SIZE = 16
const LEGEND_TITLE_SIZE = 18

colors    = [colorant"#1f77b4", colorant"#2ca02c", colorant"#d62728"]   # one per z
unit_conv = ustrip(auconvert(u"u", 1) / auconvert(u"ps", 1))            # au friction → u⋅ps⁻¹

const N_PANELS = length(R_VALUES)

fig = Figure(size=(HokseonPlots.TWO_COLUMN_WIDTH, HokseonPlots.TWO_COLUMN_WIDTH * 1.3),
             figure_padding=(2, 6, 2, 2),
             fonts=(;regular=MINION_FONT))

# Panels live in a nested grid so they can be glued together (rowgap = 0)
# without affecting the legend / shared y-label in the outer layout.
panel_grid = GridLayout(fig[1, 2])
axes = Vector{Any}(undef, N_PANELS)

for p in 1:N_PANELS
    is_bottom = (p == N_PANELS)
    axes[p] = MyAxis(panel_grid[p, 1];
                xlabel             = is_bottom ? "ħω  (eV)" : "",
                ylabel             = "",
                xlabelsize         = AXIS_LABEL_SIZE,
                xticklabelsize     = TICK_LABEL_SIZE,
                yticklabelsize     = TICK_LABEL_SIZE,
                xticklabelsvisible = is_bottom,   # only the bottom panel labels x
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

linkxaxes!(axes...)
rowgap!(panel_grid, 0)   # attach the panels

for (p, r) in enumerate(R_VALUES)
    params_list = dict_list(Dict{String, Any}(
        "r"           => [r],
        "z"           => Z_VALUES,
        "temperature" => [TEMPERATURE],

        ## extra [] so collect(...) is treated as a single parameter
        "energy"      => [ENERGY_GRID],
    ))

    if typeof(params_list) != Vector{Dict{String, Any}}
        params_list = [params_list]
    end

    # Only z varies within a panel → deterministic z → colour mapping
    sort!(params_list, by = d -> d["z"])

    for (i, params_dict) in enumerate(params_list)
        # frequency-dependent trace-averaged friction  tr(Λ)/ndof  — solid curve
        Λ_avg = buildAveragedFrequencyLambda(params_dict) .* unit_conv

        # Markovian trace-averaged friction (scalar) — dashed horizontal line
        γ_avg = buildAveragedMarkovianFriction(params_dict) * unit_conv

        ω_eV = params_dict["energy"]   # grid is already in eV

        lines!(axes[p],  ω_eV, Λ_avg; color=colors[i], linestyle=:solid, linewidth=2)
        hlines!(axes[p], [γ_avg];     color=colors[i], linestyle=:dash,  linewidth=2)
    end

    # Per-panel fixed-parameter annotation (top-right, inside the axis).
    # The top panel also carries the global system / temperature label.
    ann = p == 1 ? "r = $r Å" : "r = $r Å"
    text!(axes[p], 0.98, 0.95;
          text  = ann,
          space = :relative, align = (:right, :top), fontsize = ANNOTATION_SIZE)
end

# Shared rotated y-label spanning all panels
Label(fig[1, 1], "⟨K(ω,x)⟩  (u⋅ps⁻¹)";
      rotation = π/2, tellheight = false, tellwidth = true,
      fontsize = AXIS_LABEL_SIZE, padding = (0, 0, 0, 0))

# Two-group horizontal legend across the top:
#   colour group → z (Å);  style group → Frequency (solid) vs Markovian (dash)
color_elems  = [LineElement(color=colors[i], linewidth=2) for i in 1:length(Z_VALUES)]
color_labels = ["$z" for z in sort(Z_VALUES)]

style_elems  = [LineElement(color=:black, linestyle=:solid, linewidth=2),
                LineElement(color=:black, linestyle=:dash,  linewidth=2)]
style_labels = ["Frequency", "Markovian"]

# Span BOTH figure columns (y-label + panels) so the stacked legends' centred
# halign references the full figure width, not just the panel column.
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

outpath = plotsdir("friction", "NOAu", "LambdaAveraged_r.pdf")
mkpath(dirname(outpath))
save(outpath, fig)
