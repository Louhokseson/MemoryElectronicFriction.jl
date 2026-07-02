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
positions  = [1.9, 2.0, 2.1, 2.2]
Γ_values   = [0.8, 0.2, 0.02]
temperature = 300
energy_grid = collect(0.000001:0.001:11.0)
impuritymodel = :ErpenbeckThossAdsorbate

# Peak marker: ħω*(x) = |h(x)| + C·k_BT. C is the Fermi-edge "confidence width" —
# how many k_BT past E_F until the Fermi factor is essentially saturated:
#   n_F(−C·k_BT) = 1/(1+e^{−C});  C=4 ⇒ ≈ 98 %.
const C_FERMI = 3
kT_eV = ustrip(auconvert(u"eV", austrip(temperature * u"K")))   # k_BT in eV

# x-axis range (eV) for the MyAxis `limits`. (The peak marker in this variant is
# a vlines! in DATA coordinates, so unlike the scatter-marker version it does not
# use this for a relative-space mapping.)
const XLIMITS = (-0.2, 11.0)

colors = [colorant"#1f77b4", colorant"#2ca02c", colorant"#ff7f0e", colorant"#d62728"]

# Panel labels (top-left of each stacked panel)
panel_labels = ["(a)", "(b)", "(c)"]

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

fig = Figure(size=(HokseonPlots.TWO_COLUMN_WIDTH, HokseonPlots.TWO_COLUMN_WIDTH * 1),
             figure_padding=(2, 6, 2, 2),
             fonts=(;regular=MINION_FONT))

# Build 3 stacked panels (one per Γ) sharing x-axis and y-label
axes = Axis[]
for (k, Γ) in enumerate(Γ_values)
    is_bottom = (k == length(Γ_values))

    # First pass — compute every curve for this Γ so the panel's y-top is known
    # BEFORE the axis is built. MyAxis builds a linked twin axis (for the mirror
    # ticks), so a post-hoc ylims! is fought back by the twin's auto-limits; the
    # y-top must instead be passed through the `limits` kwarg at creation (which
    # IS forwarded to both axes).
    panel_curves = NamedTuple[]
    panel_ymax   = 0.0
    panel_ymin   = Inf
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
        γ    = ones(length(energy_au)) .* ustrip.(MarkovianFriction_NQCD_au[1] .* auconvert.(u"u", 1) ./ auconvert.(u"ps", 1))
        ω_ev = ustrip.(auconvert.(u"eV", energy_au))
        Λ    = ustrip.(Λ_au .* auconvert.(u"u", 1) ./ auconvert.(u"ps", 1))

        # Peak-position marker (analytical particle–hole threshold):
        #     ħω*(x) = |h(x)| + C_FERMI·k_BT   (level ϵₐ(x)=h(x) excited across the
        # saturated Fermi edge). Drawn later as a dotted vertical line at ω*.
        h_ev   = abs(ustrip(auconvert(u"eV", HokseonReproduce.adsorbate_h(position_au, adsorbate_m))))
        ω_peak = h_ev + C_FERMI * kT_eV

        # Track the positive-value range (log axis) across all curves so the
        # y-window can place the topmost line at a fixed fraction of the panel.
        posvals    = filter(x -> isfinite(x) && x > 0, vcat(γ, Λ))
        panel_ymax = max(panel_ymax, maximum(posvals))
        panel_ymin = min(panel_ymin, minimum(posvals))
        push!(panel_curves, (i=i, ω_ev=ω_ev, γ=γ, Λ=Λ, ω_peak=ω_peak))
    end

    # Size the log y-window so the TOPMOST line sits at a fixed fraction r_top of
    # the panel height in EVERY panel, regardless of its dynamic range. The green
    # Markovian dash / solid peak span the full panel width, so both the top-left
    # (a)/(b)/(c) label and the top-right Δ annotation need clear space above that
    # line. With headroom = α·(decades of data) and α = (1−r)/r, the topmost line
    # lands at 1/(1+α) = r in all panels — a uniform multiplicative factor would
    # NOT, since the three panels span different numbers of decades (which is why
    # the Δ annotation collided in the b/c panels). A small margin below ymin keeps
    # the lowest point off the bottom spine.
    r_top = 0.72
    α     = (1 - r_top) / r_top
    D_dec = log10(panel_ymax / panel_ymin)        # decades spanned by the data
    y_bot = panel_ymin / 1.2
    y_top = panel_ymax * 10.0^(α * D_dec)

    ax = MyAxis(fig[k, 2];
                xlabel = is_bottom ? "ħω  (eV)" : "",
                ylabel = "",
                # Symmetric-log: LINEAR in |K| < 1 (so the high-ω tail can flatten
                # onto the 0 baseline — the convergence feature a pure log10 axis
                # cannot show, since log10(0) = -∞), LOGARITHMIC for |K| > 1 (so the
                # x=2.0 peak, ~1539 at small ω, is compressed instead of dominating).
                yscale             = log10,
                xlabelsize         = AXIS_LABEL_SIZE,
                xticklabelsize     = TICK_LABEL_SIZE,
                yticklabelsize     = TICK_LABEL_SIZE,
                limits = (XLIMITS..., y_bot, y_top),
                xgridvisible = false, ygridvisible = false,
                xticklabelsvisible = is_bottom,
                xticksvisible      = is_bottom,
                xticks             = 0:2:10,
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

    # Second pass — draw the precomputed curves for this Γ. ω* is the dotted
    # vertical peak marker (particle–hole threshold) in DATA coordinates (eV).
    for c in panel_curves
        lines!(ax, c.ω_ev, c.γ; color=colors[c.i], linestyle=:dash,  linewidth=2)
        vlines!(ax, c.ω_peak;   color=colors[c.i], linestyle=:dot,   linewidth=1.5)
        lines!(ax, c.ω_ev, c.Λ; color=colors[c.i], linestyle=:solid, linewidth=2)
    end

    # Per-panel (a)/(b)/(c) label (top-right corner, inside the axis), placed in the
    # cleared headroom via relative coordinates. Bold (PRX convention): no bold
    # MinionPro file ships with the project, so faux-bold the label by stroking the
    # glyphs in their own colour — this keeps the Minion typeface rather than
    # falling back to a generic :bold sans-serif.
    text!(ax, 0.98, 0.97; text=panel_labels[k], space=:relative, align=(:right, :top),
          fontsize=ANNOTATION_SIZE, strokewidth=0.6, strokecolor=:black)

    # Per-panel Γ annotation (centre top, inside the axis)
    text!(ax, 0.5, 0.95; text="Δ₀ = $(Γ/2) eV", space=:relative, align=(:center, :top), fontsize=ANNOTATION_SIZE)
end

# Link x-axes across the three panels and tighten vertical spacing.
# rowgap! must be set AFTER the layout is populated; 0 removes the inter-panel gap
# so the stacked panels share a single x-axis edge (PRX-style).
linkxaxes!(axes...)
rowgap!(fig.layout, 0)

# Shared y-label spanning all 3 panels (one rotated Label rather than per-axis labels)
Label(fig[1:length(Γ_values), 1], "K(ω ; x)  (u⋅ps⁻¹)";
      rotation = π/2, tellheight = false, tellwidth = true,
      fontsize = AXIS_LABEL_SIZE,
      padding = (0, 4, 0, 0))

# Horizontal legend across the TOP of the figure, spanning the axes column.
# In a PRX two-column figure a side legend wastes lateral space; a thin
# horizontal strip above the panels keeps the data area as wide as possible.
#
# The three groups are stacked one-per-row inside a nested GridLayout (rather
# than side-by-side, whose combined natural width exceeds the ~435-unit axes
# column and gets clipped). To keep the groups visually ALIGNED, titles and
# entries live in SEPARATE columns of the grid: col 1 holds the right-aligned
# group titles (so they share a common right edge) and col 2 holds the
# left-aligned entries (so every row's entries share a common left edge).
# A nested GridLayout isolates the legend's row spacing from the panels'
# rowgap!(fig.layout, 0).
color_elems  = [LineElement(color=colors[i], linewidth=2) for i in 1:length(positions)]
color_labels = ["$(p)" for p in positions]

style_elems  = [LineElement(color=:black, linestyle=:solid, linewidth=2),
                LineElement(color=:black, linestyle=:dash,  linewidth=2)]
style_labels = ["Frequency", "Markovian"]

# Span BOTH columns (y-label + axes) so the legend is centred over the WHOLE
# figure width rather than just the axes column (column 2). tellwidth=true makes
# the nested grid shrink-wrap its content and halign=:center then centres that
# block on the full figure.
legend_grid = GridLayout(fig[0, 1:2]; halign = :center, tellwidth = true)

# Shared settings for the per-row entry legends. Titles are NOT drawn by the
# Legend here (they are separate Labels in col 1); halign=:left + tellwidth=true
# make each row's entries flush-left against the common entry edge in col 2.
legend_kwargs = (
    orientation   = :horizontal,
    nbanks        = 1,
    tellwidth     = true,
    tellheight    = true,
    halign        = :left,
    framevisible  = false,
    patchsize     = (26, 12),
    colgap        = 8,            # gap between entries within a row
    patchlabelgap = 4,
    padding       = (2, 2, 1, 1),
    labelsize     = LEGEND_LABEL_SIZE,
    labelfont     = MINION_FONT,
)

# Peak-marker key: the dotted vertical lines sit at the particle–hole threshold
# for taking the level ϵₐ(x)=h(x) across the Fermi sea — at |h(x)| plus the
# Fermi-edge saturation width C·k_BT (n_F→1), i.e. ħω = |h(x)| + C·k_BT.
peak_elems  = [LineElement(points=[Point2f(0.5, 0), Point2f(0.5, 1)], color=:gray25, linestyle=:dot, linewidth=1.5)]
peak_labels = ["ħω = |h(x)| + $(C_FERMI) kʙT"]

# Column 1: group titles as right-aligned Labels (so all three share a common
# right edge). Column 2: the entries, left-aligned. col 1 sizes to the widest
# title ("Friction") and col 2 to the widest entry row ("x  (Å)") — this is
# what visually aligns the three groups.
legend_titles = ["x  (Å)", "Friction", "Peak"]
for (r, t) in enumerate(legend_titles)
    Label(legend_grid[r, 1], t; halign = :right,
          fontsize = LEGEND_TITLE_SIZE, font = MINION_FONT)
end

Legend(legend_grid[1, 2], color_elems, color_labels; legend_kwargs...)
Legend(legend_grid[2, 2], style_elems, style_labels; legend_kwargs...)
Legend(legend_grid[3, 2], peak_elems,  peak_labels;  legend_kwargs...)

colgap!(legend_grid, 8)   # gap between the title column and the entry column
rowgap!(legend_grid, 2)

colgap!(fig.layout, 4)


display(fig)

save(plotsdir("friction", "ErpenbeckThoss", "Lambda_Gamma_log_vertical.pdf"), fig)
