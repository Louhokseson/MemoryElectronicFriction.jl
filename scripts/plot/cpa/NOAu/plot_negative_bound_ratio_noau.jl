# plot_negative_bound_ratio_noau.jl — size of the negative-tail bound relative
# to the memory-CPA energy loss, against incidence energy; one panel per
# initial vibrational state.
#
#     ratio(Eₜ) = |⟨ΔE_neg⟩| / ⟨ΔE_mem⟩,   ⟨·⟩ = mean over the trajectories
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
# Both read the diag(μ, M) MD (mass [μ, M], Eₜ = 0.2–1.0 eV). The memory CPA
# at 300 K is run with two kernel averages, `arithmetic` and `endpoint`;
# KERNEL_AVGS selects which are drawn (default `endpoint` only). Energies
# without a memory-CPA file are skipped, and the script stops before the
# figure if there is none at all. Each panel has its own y scale (from 0).
#
# Which ΔE_neg results (same option as run_negative_bound_CPA_noau.jl):
#   julia --project plot_negative_bound_ratio_noau.jl               # data/sims/cpa/NOAu/negative_bound
#   julia --project plot_negative_bound_ratio_noau.jl --tag <name>  # .../negative_bound_<name>
# The old_engine set (a(q) from before the quadrature fix) exists only for the
# diag(μ, μ) MD, now in data/sims/cpa/NOAu_zmass_mu, so it no longer resolves.
#
# Output: plots/friction/NOAu/negative/negative_bound_ratio_vib<v…>[_<TAG>].{pdf,png}

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

const VIBS        = [0, 3, 16]                 # initial vibrational states, one panel each
const TAG         = let i = findfirst(==("--tag"), ARGS)   # "" → current results
                        i === nothing || i == length(ARGS) ? "" : ARGS[i + 1]
                    end
const T_K         = 300
const EK_LIST_eV  = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # the diag(μ, M) MD grid
const KERNEL_AVGS = [:endpoint]                # memory-CPA variant(s); add :arithmetic to overlay it

# MD parameters: must match run_md.jl so the CPA filenames resolve.
md_params(vib, Ek) = Dict{String, Any}(
    "mass"                  => [14.007 * 15.999 / (14.007 + 15.999), 14.007 + 15.999] .* u"u",   # [μ_NO, M_NO]: r carries μ, z carries M
    "r0"                    => [1.15u"Å", 5.0u"Å"],
    "translational_kinetic" => Ek * u"eV",
    "state"                 => 1,
    "dt"                    => 0.25u"fs",
    "vibrational_state"     => vib,
    "trajectories"          => 1000,
)

# Same configuration as run_negative_bound_CPA_noau.jl (defaults, --tag TAG).
const BOUND_config = Dict{String, Any}(
    "model"        => :NOAu,
    "T_K"          => T_K,
    "omega_max_eV" => nyquist_omega_max_eV(0.25u"fs"),
    "domega_eV"    => DEFAULT_DOMEGA_eV,
    "stride"       => 1,
    "variant"      => isempty(TAG) ? "negative_bound" : "negative_bound_$(TAG)",
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

# ratio_pct[(vib, kernel)] = (Eₜ values, |⟨ΔE_neg⟩|/⟨ΔE_mem⟩ in %)
ratio_pct = Dict((v, k) => (Float64[], Float64[]) for v in VIBS for k in KERNEL_AVGS)

for vib in VIBS
    println("\nv = $vib, T = $T_K K  (meV)", isempty(TAG) ? "" : "  [$TAG]")
    println("  Eₜ   ⟨ΔE_neg⟩  ", join([rpad("⟨ΔE_mem⟩ $k", 22) for k in KERNEL_AVGS]), "ratio (%)")
    for Ek in EK_LIST_eV
        p      = md_params(vib, Ek)
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
            push!(ratio_pct[(vib, k)][1], Ek); push!(ratio_pct[(vib, k)][2], r)
            push!(cols, rpad(@sprintf("%.2f", mean(ΔEm)), 22)); push!(ratios, @sprintf("%.3f", r))
        end
        @printf("  %3.1f  %7.3f   %s%s\n", Ek, mean(ΔE_neg), join(cols), join(ratios, " / "))
    end
end

if all(isempty(first(v)) for v in values(ratio_pct))
    error("no memory-CPA file with kernel_average ∈ $(KERNEL_AVGS) for any (v, Eₜ) in " *
          "$(datadir("sims", "cpa", "NOAu", "memory")) — nothing to plot")
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# Styling constants copied from scripts/plot/fig_3/plot_fig_3.jl. Each panel
# is about as wide as a Fig. 3 panel, so text-to-panel proportions match.
const FONT = projectdir("fonts", "dejavu-sans.book.ttf")

const FIGURE_WIDTH = 996                       # Fig. 3 full width
const PANEL_WIDTH  = FIGURE_WIDTH ÷ 3
const FIG_WIDTH    = PANEL_WIDTH * length(VIBS)
const FIG_HEIGHT   = round(Int, FIGURE_WIDTH * 0.4)

const panel_spinewidth = 1.8
const LINE_WIDTH       = 2.5
const MARKER_SIZE      = 12
const PANEL_COLGAP     = 18

const AXIS_LABEL_SIZE   = 18
const TITLE_SIZE        = 16
const TICK_LABEL_SIZE   = 16
const ANNOTATION_SIZE   = 16
const LEGEND_LABEL_SIZE = 16

const COLOR = colorant"#1f77b4"                # first Fig. 3 colour
const STYLE = Dict(:endpoint   => (marker = :circle, filled = true,  linestyle = :solid),
                   :arithmetic => (marker = :rect,   filled = false, linestyle = :dash))
const LABEL = Dict(:endpoint => "Memory CPA (endpoint)", :arithmetic => "Memory CPA (arithmetic)")

# x ticks on the Eₜ grid, shared by all panels: labels every 0.2 eV and a
# minor tick between, so every 0.1 eV data point sits on a tick. Labels every
# 1 eV once data reach 2–5 eV.
Ek_max   = maximum(maximum(first(v); init = 0.0) for v in values(ratio_pct))
Ek_step  = Ek_max <= 1.0 ? 0.2 : 1.0
Ek_ticks = Ek_step:Ek_step:(ceil(Ek_max / Ek_step) * Ek_step)

# y from 0 to 1.3 × the panel's largest value: room for the panel label, as in
# Fig. 3 (an autolimit margin would scale with the data range, ~0 for flat curves).
y_top(vib) = 1.3 * maximum(maximum(ratio_pct[(vib, k)][2]; init = 0.0) for k in KERNEL_AVGS)

fig = Figure(size = (FIG_WIDTH, FIG_HEIGHT), figure_padding = (2, 10, 2, 6),
             fonts = (; regular = FONT))
panel_grid = GridLayout(fig[2, 2])

axes = map(enumerate(VIBS)) do (col, vib)
    Axis(panel_grid[1, col];
         title              = "v = $vib",
         titlesize          = TITLE_SIZE,
         titlefont          = FONT,
         xticklabelsize     = TICK_LABEL_SIZE,
         yticklabelsize     = TICK_LABEL_SIZE,
         limits             = (Ek_step / 2 + 0.05, last(Ek_ticks) + 0.05, 0, y_top(vib) > 0 ? y_top(vib) : nothing),
         xticks             = Ek_ticks,
         xtickformat        = v -> [string(round(x; digits = 1)) for x in v],
         xminorticks        = (first(Ek_ticks) + Ek_step / 2):Ek_step:last(Ek_ticks),
         xminorticksvisible = true,
         xminortickwidth    = panel_spinewidth,
         xminorticksize     = 4,
         xticksize          = 8,
         yticksize          = 8,
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
linkxaxes!(axes...)
colgap!(panel_grid, PANEL_COLGAP)

for (col, vib) in enumerate(VIBS), k in KERNEL_AVGS
    x, y = ratio_pct[(vib, k)]
    isempty(x) && continue
    s = STYLE[k]
    lines!(axes[col], x, y; color = COLOR, linewidth = LINE_WIDTH, linestyle = s.linestyle)
    scatter!(axes[col], x, y; marker = s.marker, markersize = MARKER_SIZE,
             color = s.filled ? COLOR : :white, strokecolor = s.filled ? :white : COLOR,
             strokewidth = s.filled ? 1.2 : 1.8,
             label = LABEL[k])
end

# Panel labels (a), (b), (c), faux-bold as in Fig. 3.
for (col, ax) in enumerate(axes)
    text!(ax, 0.04, 0.95; text = "($('a' + col - 1))", space = :relative,
          align = (:left, :top), fontsize = ANNOTATION_SIZE,
          strokewidth = 0.4, strokecolor = :black)
end

# Shared axis labels, as in Fig. 3.
Label(fig[2, 1], rich("|⟨ΔE", subscript("neg"), "⟩| / ⟨ΔE", subscript("mem"), "⟩  (%)");
      rotation = π / 2, tellheight = false, tellwidth = true, fontsize = AXIS_LABEL_SIZE)
Label(fig[3, 2], rich("E", subscript("t"), "  (eV)");
      tellwidth = false, tellheight = true, fontsize = AXIS_LABEL_SIZE)

# Header: conditions (one series) or a legend (several memory-CPA variants).
header = "T = $T_K K" * (length(KERNEL_AVGS) == 1 ? ",  $(LABEL[only(KERNEL_AVGS)])" : "")
Label(fig[1, 2], header; fontsize = ANNOTATION_SIZE, tellwidth = false)
length(KERNEL_AVGS) > 1 &&
    axislegend(axes[end]; position = (:right, :center), framevisible = false,
               labelsize = LEGEND_LABEL_SIZE, patchsize = (18, 10), rowgap = 0)

colgap!(fig.layout, 4)
rowgap!(fig.layout, 4)

display(fig)

outpath = plotsdir("friction", "NOAu", "negative",
                   "negative_bound_ratio_vib$(join(VIBS, '-'))" * (isempty(TAG) ? "" : "_$(TAG)") * ".pdf")
mkpath(dirname(outpath))
save(outpath, fig)
save(replace(outpath, ".pdf" => ".png"), fig; px_per_unit = 2)
@info "Saved" outpath
