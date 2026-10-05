# plot_negative_bound_ratio_noau.jl — size of the negative-tail bound relative
# to the memory-CPA energy loss, against incidence energy.
#
#     ratio(Eₖ) = |⟨ΔE_neg⟩| / ⟨ΔE_mem⟩,   ⟨·⟩ = mean over the trajectories
#                                            (a ratio of means, not a mean of ratios),
#
# with ΔE_neg ≡ ΔE_a = Δt Σᵢ a(qᵢ)|vᵢ|² (the isotropic bound of
# docs/negative_tail_markovian_bound.md, from
# scripts/compute/cpa/NOAu/run_negative_bound_CPA_noau.jl) and ΔE_mem the
# memory CPA (scripts/compute/cpa/NOAu/run_memory_CPA_noau.jl), both summed
# over the r and z rows of DeltaE_au. ΔE_neg is the *bound* ΔE_a ≤ ΔE₋ ≤ 0 of
# the note, not the exact negative-tail contribution ΔE₋: −ΔE_neg is the most
# energy the negative tail of Λ(ω) can return, so the ratio bounds its share
# of the energy loss.
#
# The memory CPA at 300 K exists with two kernel averages: `endpoint` for
# Eₖ = 0.2–1.0 eV and `arithmetic` for 0.2, 0.5, 1.0, 4.0 eV (they agree to
# < 0.5 % where both exist). KERNEL_AVGS selects which are drawn; the default
# is `endpoint` only. Energies without a memory-CPA file are skipped.
#
# Output: plots/friction/NOAu/negative/negative_bound_ratio_vib<v>.{pdf,png}

using Distributed
using DrWatson
@quickactivate "MemoryElectronicFriction"

using HDF5
using Printf
using Statistics: mean
using Unitful, UnitfulAtomic
using CairoMakie
using Colors

using MemoryElectronicFriction

# Search defaults of the a(q) run (DEFAULT_DOMEGA_eV, nyquist_omega_max_eV),
# so the ΔE_neg filenames resolve exactly as the compute scripts wrote them.
include("../../../compute/friction/NOAu/NegativeEigenvalue.jl")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const VIB         = 0                          # initial vibrational state
const T_K         = 300
const EK_LIST_eV  = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
const KERNEL_AVGS = [:endpoint]                # memory-CPA variant(s); add :arithmetic to overlay it

# MD parameters: must match run_md.jl so the CPA filenames resolve.
md_params(Ek) = Dict{String, Any}(
    "mass"                  => (14.007 * 15.999 / (14.007 + 15.999)) * u"u",
    "r0"                    => [1.15u"Å", 5.0u"Å"],
    "translational_kinetic" => Ek * u"eV",
    "state"                 => 1,
    "dt"                    => 0.25u"fs",
    "vibrational_state"     => VIB,
    "trajectories"          => 1000,
)

# Same configuration as run_negative_bound_CPA_noau.jl (defaults).
const BOUND_config = Dict{String, Any}(
    "model"        => :NOAu,
    "T_K"          => T_K,
    "omega_max_eV" => nyquist_omega_max_eV(0.25u"fs"),
    "domega_eV"    => DEFAULT_DOMEGA_eV,
    "stride"       => 1,
    "variant"      => "negative_bound",
)
memory_config(kernel_average) = Dict{String, Any}(
    "model" => :NOAu, "T_K" => T_K, "stride" => 1, "kernel_average" => kernel_average)

# ---------------------------------------------------------------------------
# Data: per-trajectory total ΔE (r + z rows) in meV, or nothing if missing.
# ---------------------------------------------------------------------------

const au_to_meV = 1e3 * ustrip(auconvert(u"eV", 1))

function total_delta_energy_meV(p, cfg)
    path = datadir(CPA_dict_to_data_savename(p, cfg)...)
    isfile(path) || return nothing
    return vec(sum(h5read(path, "DeltaE_au"); dims = 1)) .* au_to_meV
end

# ratio_pct[kernel] = (Eₖ values, |⟨ΔE_neg⟩|/⟨ΔE_mem⟩ in %)
ratio_pct = Dict(k => (Float64[], Float64[]) for k in KERNEL_AVGS)

println("v = $VIB, T = $T_K K  (meV)")
println("  Eₖ   ⟨ΔE_neg⟩  ", join([rpad("⟨ΔE_mem⟩ $k", 22) for k in KERNEL_AVGS]), "ratio (%)")
for Ek in EK_LIST_eV
    p   = md_params(Ek)
    ΔE_neg = total_delta_energy_meV(p, BOUND_config)
    ΔE_neg === nothing && continue
    cols = String[]; ratios = String[]
    for k in KERNEL_AVGS
        ΔEm = total_delta_energy_meV(p, memory_config(k))
        if ΔEm === nothing || length(ΔEm) != length(ΔE_neg)
            push!(cols, rpad("–", 22)); push!(ratios, "–")
            continue
        end
        r = 100 * abs(mean(ΔE_neg)) / mean(ΔEm)
        push!(ratio_pct[k][1], Ek); push!(ratio_pct[k][2], r)
        push!(cols, rpad(@sprintf("%.2f", mean(ΔEm)), 22)); push!(ratios, @sprintf("%.3f", r))
    end
    @printf("  %3.1f  %7.3f   %s%s\n", Ek, mean(ΔE_neg), join(cols), join(ratios, " / "))
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# Styling constants copied from scripts/plot/fig_3/plot_fig_3.jl. One panel
# at half the Fig. 3 width keeps the text-to-panel proportions of Fig. 3.
const FONT = projectdir("fonts", "dejavu-sans.book.ttf")

const FIGURE_WIDTH = 996                       # Fig. 3 full width
const FIG_WIDTH    = FIGURE_WIDTH ÷ 2
const FIG_HEIGHT   = round(Int, FIG_WIDTH * 0.75)

const panel_spinewidth = 1.8
const LINE_WIDTH       = 2.5
const MARKER_SIZE      = 12

const AXIS_LABEL_SIZE   = 18
const TICK_LABEL_SIZE   = 16
const ANNOTATION_SIZE   = 16
const LEGEND_LABEL_SIZE = 16

const COLOR = colorant"#1f77b4"                # first Fig. 3 colour
const STYLE = Dict(:endpoint   => (marker = :circle, filled = true,  linestyle = :solid),
                   :arithmetic => (marker = :rect,   filled = false, linestyle = :dash))
const LABEL = Dict(:endpoint => "Memory CPA (endpoint)", :arithmetic => "Memory CPA (arithmetic)")

# x ticks on the Eₖ grid: labels every 0.2 eV and a minor tick between, so every
# 0.1 eV data point sits on a tick. Labels every 1 eV once data reach 2–5 eV.
Ek_max     = maximum(maximum(first(v); init = 0.0) for v in values(ratio_pct))
Ek_step    = Ek_max <= 1.0 ? 0.2 : 1.0
Ek_ticks   = Ek_step:Ek_step:(ceil(Ek_max / Ek_step) * Ek_step)

fig = Figure(size = (FIG_WIDTH, FIG_HEIGHT), figure_padding = (2, 10, 2, 6),
             fonts = (; regular = FONT))

ax = Axis(fig[1, 1];
          xlabel             = rich("E", subscript("t"), "  (eV)"),
          ylabel             = rich("|⟨ΔE", subscript("neg"), "⟩| / ⟨ΔE", subscript("mem"), "⟩  (%)"),
          xlabelsize         = AXIS_LABEL_SIZE,
          ylabelsize         = AXIS_LABEL_SIZE,
          xticklabelsize     = TICK_LABEL_SIZE,
          yticklabelsize     = TICK_LABEL_SIZE,
          limits             = (Ek_step / 2 + 0.05, last(Ek_ticks) + 0.05, 0, nothing),
          xticks             = Ek_ticks,
          xtickformat        = v -> [string(round(x; digits = 1)) for x in v],
          xminorticks        = (first(Ek_ticks) + Ek_step / 2):Ek_step:last(Ek_ticks),
          xminorticksvisible = true,
          xminortickwidth    = panel_spinewidth,
          xminorticksize     = 4,
          xticksize          = 8,
          yticksize          = 8,
          yautolimitmargin   = (0.0, 0.15),
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

for k in KERNEL_AVGS
    x, y = ratio_pct[k]
    isempty(x) && continue
    s = STYLE[k]
    lines!(ax, x, y; color = COLOR, linewidth = LINE_WIDTH, linestyle = s.linestyle)
    scatter!(ax, x, y; marker = s.marker, markersize = MARKER_SIZE,
             color = s.filled ? COLOR : :white, strokecolor = s.filled ? :white : COLOR,
             strokewidth = s.filled ? 1.2 : 1.8,
             label = LABEL[k])
end

# One series: name the memory-CPA variant in the corner note, no legend box.
corner = length(KERNEL_AVGS) == 1 ? "v = $VIB,  T = $T_K K\n$(LABEL[only(KERNEL_AVGS)])" :
                                    "v = $VIB,  T = $T_K K"
text!(ax, 0.96, 0.95; text = corner, space = :relative, align = (:right, :top),
      justification = :right, fontsize = ANNOTATION_SIZE)
length(KERNEL_AVGS) > 1 &&
    axislegend(ax; position = (:right, :center), framevisible = false,
               labelsize = LEGEND_LABEL_SIZE, patchsize = (18, 10), rowgap = 0)

display(fig)

outpath = plotsdir("friction", "NOAu", "negative", "negative_bound_ratio_vib$(VIB).pdf")
mkpath(dirname(outpath))
save(outpath, fig)
save(replace(outpath, ".pdf" => ".png"), fig; px_per_unit = 2)
@info "Saved" outpath
