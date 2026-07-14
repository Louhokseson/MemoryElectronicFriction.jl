## Only for the main process
using Distributed
using DrWatson
@quickactivate "HokseonReproduce" ## Activate project everywhere
import Pkg; Pkg.precompile() ## Precompile packages in master to speed up workers' precompilation
using HDF5
using DelimitedFiles
using HokseonAssistant
using CairoMakie
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
@everywhere using NQCModels.QuantumModels
@everywhere using NQCModels


function buildSystemBath(params_dict::Dict{String, Any})
    @unpack  position, impuritymodel, temperature, energy = params_dict

    if impuritymodel == :ErpenbeckThossAdsorbate
        Γ = params_dict["Γ"]
        adsorbate_m = ErpenbeckThossAdsorbate(Γ=austrip(Γ*u"eV"))
    else
        adsorbate_m = eval(impuritymodel)()
    end


    energy_au = austrip.(energy*u"eV")
    temperature_au = austrip.(temperature*u"K")
    position_au = austrip.(position*u"Å")

    return adsorbate_m, position_au, energy_au, temperature_au
end

function NQCD_MarkovianFriction(params_dict::Dict{String, Any})
    @unpack  position, impuritymodel, temperature = params_dict

    temperature_au = austrip.(temperature*u"K")
    position_au = austrip.(position*u"Å")

    if impuritymodel == :ErpenbeckThossAdsorbate
        Γ = params_dict["Γ"]

        adsorbate_m = NQCModels.ErpenbeckThoss(;Γ=austrip(Γ*u"eV"))
        M = 1000
        bw = austrip(100u"eV")
        model = WideBandBath(adsorbate_m; step=(2bw)/M, bandmin=-bw, bandmax=bw)
        atoms = Atoms(1u"u")
        sim = Simulation{DiabaticMDEF}(atoms, model, friction_method=NQCCalculators.WideBandExact(model.ρ, 1/temperature_au))

        MarkovianFriction = [only(NQCCalculators.evaluate_friction(sim.cache, hcat(x))) for x in position_au]
        return MarkovianFriction
    end
end


# Screening parameters
positions  = [1.8, 1.9, 2.1, 2.2]
Γ_values   = [0.8, 0.2, 0.02]
temperature = 300
energy_grid = collect(0.000001:0.001:30.0)
impuritymodel = :ErpenbeckThossAdsorbate

colors = [colorant"#1f77b4", colorant"#2ca02c", colorant"#ff7f0e", colorant"#d62728"]

const panel_spinewidth = 1.8
const MINION_FONT = projectdir("fonts", "MinionPro-Capt.otf")

# PRX-standard font sizes for the 481-unit (TWO_COLUMN_WIDTH) canvas.
# A clean 3-step hierarchy that stays legible at column width while fitting
# the horizontal legend and avoiding y-tick crowding (rowgap! = 0):
#   primary   (24) — axis labels (carry units, must survive reduction)
#   secondary (22) — panel annotation + legend group titles
#   data      (20) — numeric tick labels + legend entries (kept equal)
const AXIS_LABEL_SIZE   = 24
const TICK_LABEL_SIZE   = 20
const ANNOTATION_SIZE   = 22
const LEGEND_LABEL_SIZE = 16
const LEGEND_TITLE_SIZE = 18

fig = Figure(size=(HokseonPlots.TWO_COLUMN_WIDTH, HokseonPlots.TWO_COLUMN_WIDTH * 0.8),
             figure_padding=(2, 6, 2, 2),
             fonts=(;regular=MINION_FONT))

# Build 3 stacked panels (one per Γ) sharing x-axis and y-label
axes = Axis[]
for (k, Γ) in enumerate(Γ_values)
    is_bottom = (k == length(Γ_values))
    ax = MyAxis(fig[k, 2];
                xlabel = is_bottom ? "ħω  (eV)" : "",
                ylabel = "",
                xlabelsize         = AXIS_LABEL_SIZE,
                xticklabelsize     = TICK_LABEL_SIZE,
                yticklabelsize     = TICK_LABEL_SIZE,
                limits = (0.0, 7.0, nothing, nothing),
                xgridvisible = false, ygridvisible = false,
                xticklabelsvisible = is_bottom,
                xticksvisible      = is_bottom,
                xticksmirrored     = false,
                yticksmirrored     = true,   # mirror y-ticks onto right spine
                xtickalign         = 1,      # x-ticks point INTO the axis
                ytickalign         = 1,      # y-ticks point INTO the axis
                xminortickalign    = 1,
                yminortickalign    = 1,
                spinewidth         = panel_spinewidth,
                xtickwidth         = panel_spinewidth,
                ytickwidth         = panel_spinewidth)

    # PRX-style: when stacking shared-x panels, the upper panels must have
    # ALL their x-decorations removed (ticks, mirror ticks, labels) so the
    # adjacent panel edges visually merge into a single shared axis line.
    # `xticksmirrored=false` already kills the top mirror ticks of the bottom
    # panel; for upper panels we additionally hide their bottom ticks.
    if !is_bottom
        hidexdecorations!(ax; ticks=true, ticklabels=true, label=true,
                              minorticks=true, grid=false)
    end

    # Defensive observable-level override.
    # MyAxis (HokseonPlots wrapper) ignores `xticksmirrored=false` passed as a
    # kwarg — the previous PDF showed mirror tick stubs at the top edge of the
    # bottom panel. Setting the observable directly forces it off.
    ax.xticksmirrored[]     = false
    ax.xminorticksvisible[] = false   # also kill any minor (mirror) ticks

    # MyAxis draws mirror ticks via a TWIN axis (xaxisposition=:top). Setting
    # xticksmirrored on `ax` only affects the main axis; the twin still renders
    # top-edge ticks whenever `xticksvisible=true` (i.e. the bottom panel).
    # Hide x-decorations on every twin axis sharing this cell so no stubs
    # appear at the boundary between middle and bottom panels.
    for content in contents(fig[k, 2])
        if content !== ax && content isa Axis
            hidexdecorations!(content; ticks=true, minorticks=true,
                              ticklabels=true, label=true, grid=false)
        end
    end

    # Hide the TOP spine of every panel except the topmost one. This collapses
    # the otherwise-double 2×1.8 boundary into a single 1.8-wide line drawn by
    # the spine above, and removes the anchor for any residual top-edge tick
    # rendering on the bottom panel (mirror stubs reported in Lambda_Gamma.pdf).
    if k > 1
        ax.topspinevisible[] = false
    end

    push!(axes, ax)

    for (i, pos) in enumerate(positions)
        params_dict = Dict{String, Any}(
            "impuritymodel" => impuritymodel,
            "centre"        => 0,
            "position"      => pos,
            "temperature"   => temperature,
            "energy"        => energy_grid,
            "Γ"             => Γ,
        )

        MarkovianFriction_NQCD_au = NQCD_MarkovianFriction(params_dict)

        adsorbate_m, position_au, energy_au, temperature_au = buildSystemBath(params_dict)

        Λ_au = FrequencyLambda.Lambda.(energy_au, Ref(adsorbate_m), Ref(position_au), Ref(temperature_au))

        γ = ones(length(energy_au)) .* ustrip.(MarkovianFriction_NQCD_au[1] .* auconvert.(u"u", 1) ./ auconvert.(u"ps", 1))

        ω_ev = ustrip.(auconvert.(u"eV", energy_au))
        Λ    = ustrip.(Λ_au .* auconvert.(u"u", 1) ./ auconvert.(u"ps", 1))

        lines!(ax, ω_ev, Λ; color=colors[i], linestyle=:solid, linewidth=2)
        lines!(ax, ω_ev, γ; color=colors[i], linestyle=:dash,  linewidth=2)
    end

    # Per-panel Γ annotation (top-right, inside the axis)
    text!(ax, 0.98, 0.95; text="Γ = $(Γ) eV", space=:relative, align=(:right, :top), fontsize=ANNOTATION_SIZE)
end

# Link x-axes across the three panels and tighten vertical spacing.
# rowgap! must be set AFTER the layout is populated; 0 removes the inter-panel gap
# so the stacked panels share a single x-axis edge (PRX-style).
linkxaxes!(axes...)
rowgap!(fig.layout, 0)

# Shared y-label spanning all 3 panels (one rotated Label rather than per-axis labels)
Label(fig[1:length(Γ_values), 1], "K(x ; ω)  (u⋅ps⁻¹)";
      rotation = π/2, tellheight = false, tellwidth = true,
      fontsize = AXIS_LABEL_SIZE,
      padding = (0, 4, 0, 0))

# Horizontal legend across the TOP of the figure, spanning the axes column.
# In a PRX two-column figure a side legend wastes lateral space; a thin
# horizontal strip above the panels keeps the data area as wide as possible.
#
# The two groups are stacked on TWO rows (inside a nested GridLayout) rather
# than placed side-by-side. Side-by-side, the combined natural width
# ("x (Å)" + 4 entries + "Friction" + "Frequency"/"Markovian" + the patches)
# exceeds the ~435-unit axes column; with tellwidth=false Makie centres the
# legend and CLIPS the edge entries (the bug seen in the PDF). One group per
# row keeps each strip well under the column width, so nothing is clipped.
# A nested GridLayout isolates the legend's row spacing from the panels'
# rowgap!(fig.layout, 0).
color_elems  = [LineElement(color=colors[i], linewidth=2) for i in 1:length(positions)]
color_labels = ["$(p)" for p in positions]

style_elems  = [LineElement(color=:black, linestyle=:solid, linewidth=2),
                LineElement(color=:black, linestyle=:dash,  linewidth=2)]
style_labels = ["Frequency", "Markovian"]

legend_grid = GridLayout(fig[0, 2])

# Shared settings for both single-row legends. `titleposition=:left` puts the
# group title on the same line as its entries, so each group is one compact row.
legend_kwargs = (
    orientation   = :horizontal,
    titleposition = :left,
    nbanks        = 1,
    tellwidth     = false,
    tellheight    = true,
    halign        = :center,
    framevisible  = false,
    patchsize     = (26, 12),
    colgap        = 8,            # gap between entries within a row
    titlegap      = 8,           # gap between group title and its entries
    patchlabelgap = 4,
    padding       = (2, 2, 1, 1),
    labelsize     = LEGEND_LABEL_SIZE,
    titlesize     = LEGEND_TITLE_SIZE,
    labelfont     = MINION_FONT,
    titlefont     = MINION_FONT,
)

Legend(legend_grid[1, 1], color_elems, color_labels, "x  (Å)";   legend_kwargs...)
Legend(legend_grid[2, 1], style_elems, style_labels, "Friction"; legend_kwargs...)
rowgap!(legend_grid, 2)

colgap!(fig.layout, 4)

display(fig)

save(plotsdir("friction", "ErpenbeckThoss", "Lambda_Gamma.pdf"), fig)