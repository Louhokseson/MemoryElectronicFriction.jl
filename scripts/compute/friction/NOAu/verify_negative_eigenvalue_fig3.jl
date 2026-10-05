# verify_negative_eigenvalue_fig3.jl — check the negative-eigenvalue engine
# against panels (d) and (h) of Fig. 3.
#
# Fig. 3 (scripts/plot/fig_3/plot_fig_3.jl) plots λ_min(ω) = eigmin Λ(ω; q) at
# 300 K for six configurations, r ∈ {1.17, 1.6} Å × z ∈ {1.7, 2.0, 3.0} Å, from
# the Λ(ω) matrices stored in figure_data/fig_3/fig_3_data.h5 (0.01–20 eV,
# 0.01 eV grid). Here the engine of run_negative_eigenvalue_noau.jl
# (NegativeEigenvalue.jl, production defaults: 0.25 eV coarse scan + Brent,
# ω ≤ ħπ/Δt = 8.27 eV) recomputes Λ at those configurations and locates the
# minimum. Each engine minimum should sit on the bottom of its stored curve.
#
# Output: a comparison table on stdout and
#   plots/friction/NOAu/negative/verify_negative_eigenvalue_fig3.{pdf,png}
#     left column   panels (d), (h) redrawn (log ω) with the engine minima
#     right column  the negative dip (linear ω) with the engine's coarse scan
#
#   julia --project -t 6 scripts/compute/friction/NOAu/verify_negative_eigenvalue_fig3.jl

using Distributed
using DrWatson
@quickactivate "MemoryElectronicFriction"

using MemoryElectronicFriction
using Unitful, UnitfulAtomic
using HDF5
using CairoMakie
using Colors
using Printf
using LinearAlgebra: eigmin

include("NegativeEigenvalue.jl")

# ---------------------------------------------------------------------------
# Engine settings: the defaults of run_negative_eigenvalue_noau.jl.
# ---------------------------------------------------------------------------

const DOMEGA_eV    = DEFAULT_DOMEGA_eV
const OMEGA_MAX_eV = nyquist_omega_max_eV(0.25u"fs")   # NO/Au MD step
const OMEGA_TOL_eV = DEFAULT_OMEGA_TOL_eV

# Pass criteria against the 0.01 eV Fig. 3 grid. The minimum is flat, so ω*
# is looser than λ_min.
const RTOL_LAMBDA = 1e-4
const ATOL_OMEGA_eV = 0.05

to_eV(ω) = ustrip(auconvert(u"eV", ω))

# ---------------------------------------------------------------------------
# Fig. 3 data: λ_min(ω) of the stored Λ(ω) matrices, in au.
# ---------------------------------------------------------------------------

function load_fig3(path)
    h5open(path, "r") do h5
        R, Z = read(h5["R_VALUES"]), sort(read(h5["Z_VALUES"]))
        λ = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
        for r in R, z in Z
            M = read(h5["r_$(r)"]["z_$(z)"]["Lambda_mats"])   # (N_ω, 2, 2)
            λ[(r, z)] = [eigmin(Symmetric(M[j, :, :])) for j in axes(M, 1)]
        end
        (; R, Z, λ, E = read(h5["ENERGY_GRID"]), T_K = read(h5["TEMPERATURE"]),
           unit_conv = read(h5["unit_conv"]))
    end
end

fig3 = load_fig3(projectdir("figure_data", "fig_3", "fig_3_data.h5"))

# ---------------------------------------------------------------------------
# Engine at the six configurations (fresh Λ from the current code).
# ---------------------------------------------------------------------------

ads      = NOAuAdsorbate()
T_au     = austrip(fig3.T_K * u"K")
ω_coarse = coarse_omega_grid(DOMEGA_eV, OMEGA_MAX_eV)
ω_tol    = austrip(OMEGA_TOL_eV * u"eV")

engine = Dict(map([(r, z) for r in fig3.R for z in fig3.Z]) do (r, z)
    q = SA[austrip(r * u"Å"), austrip(z * u"Å")]
    λ_star, ω_star, e = most_negative_eigenvalue(ads, q, T_au, ω_coarse, ω_tol)
    λ_scan = [eigmin_friction(ω, ads, q, T_au) for ω in ω_coarse]   # the engine's first stage
    (r, z) => (; λ_star, ω_star_eV = to_eV(ω_star), e, λ_scan)
end)

# ---------------------------------------------------------------------------
# Comparison table. The Fig. 3 reference is the grid minimum inside the same
# search window [Δω, ω_max]; the global grid minimum is flagged if it differs.
# ---------------------------------------------------------------------------

scale  = fig3.unit_conv * 1e3   # au → 10⁻³ u⋅ps⁻¹, the Fig. 3 column-4 units
window = findall(e -> DOMEGA_eV <= e <= OMEGA_MAX_eV, fig3.E)

println("\nλ_min at T = $(fig3.T_K) K, search window [$(DOMEGA_eV), $(OMEGA_MAX_eV)] eV; λ in 10⁻³ u⋅ps⁻¹")
println(rpad("r (Å)", 7), rpad("z (Å)", 7), rpad("Fig.3 λ", 11), rpad("ħω (eV)", 9),
        rpad("engine λ", 11), rpad("ħω* (eV)", 10), rpad("rel. diff", 11), rpad("|e·ẑ|", 8), "check")
all_pass = true
for r in fig3.R, z in fig3.Z
    λ_ref = fig3.λ[(r, z)]
    i_win = window[argmin(λ_ref[window])]
    i_all = argmin(λ_ref)
    res   = engine[(r, z)]
    rel   = (res.λ_star - λ_ref[i_win]) / abs(λ_ref[i_win])
    pass  = abs(rel) < RTOL_LAMBDA && abs(res.ω_star_eV - fig3.E[i_win]) < ATOL_OMEGA_eV
    global all_pass &= pass
    @printf("%-7.2f%-7.1f%-11.5f%-9.2f%-11.5f%-10.3f%-11.1e%-8.3f%s%s\n",
            r, z, λ_ref[i_win] * scale, fig3.E[i_win], res.λ_star * scale, res.ω_star_eV,
            rel, abs(res.e[2]), pass ? "PASS" : "FAIL",
            i_all == i_win ? "" : @sprintf("  (global grid min %.5f at %.2f eV)", λ_ref[i_all] * scale, fig3.E[i_all]))
end
println(all_pass ? "All six minima reproduced." : "Some minima NOT reproduced — see FAIL rows.")

# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------

const FONT    = projectdir("fonts", "dejavu-sans.book.ttf")
const COLORS  = [colorant"#1f77b4", colorant"#2ca02c", colorant"#d62728"]   # Fig. 3 z colours
# Marker shape repeats the z identity: the Fig. 3 green/red pair is not
# separable under deuteranopia.
const MARKERS = [:circle, :rect, :utriangle]
const INK     = colorant"#333333"
const OUTSIDE = (colorant"#9a9a9a", 0.18)   # ω > ω_max, outside the search window

fig = Figure(size = (980, 640), figure_padding = (6, 10, 6, 6), fonts = (; regular = FONT))

Label(fig[0, 1:2],
      "Negative-eigenvalue engine vs Fig. 3 (d), (h)  ·  T = $(round(Int, fig3.T_K)) K  ·  " *
      "search ħω ∈ [$(DOMEGA_eV), $(OMEGA_MAX_eV)] eV";
      fontsize = 15, color = INK, tellwidth = false)

axis_kwargs = (xgridvisible = false, ygridvisible = false, xtickalign = 1, ytickalign = 1,
               yticksmirrored = true, spinewidth = 1.2, xticklabelsize = 12, yticklabelsize = 12,
               xlabelsize = 13, ylabelsize = 13, titlesize = 13, titlecolor = INK)

for (row, r) in enumerate(fig3.R)
    is_bottom = row == length(fig3.R)
    λ_floor   = minimum(engine[(r, z)].λ_star for z in fig3.Z) * scale

    axL = Axis(fig[row + 1, 1]; axis_kwargs..., xscale = log10,
               limits = (0.008, 20, nothing, nothing), yautolimitmargin = (0.05, 0.12),
               title = row == 1 ? "λₘᵢₙ(ω) as in Fig. 3, log ω" : "",
               xlabel = is_bottom ? "ħω  (eV)" : "", xticklabelsvisible = is_bottom,
               ylabel = "λₘᵢₙ  (10⁻³ u⋅ps⁻¹)")
    axR = Axis(fig[row + 1, 2]; axis_kwargs...,
               limits = (0, OMEGA_MAX_eV + 0.6, 1.25 * λ_floor, -0.3 * λ_floor),
               title = row == 1 ? "Negative dip: coarse scan + Brent minimum" : "",
               xlabel = is_bottom ? "ħω  (eV)" : "", xticklabelsvisible = is_bottom)

    for ax in (axL, axR)
        vspan!(ax, OMEGA_MAX_eV, 20; color = OUTSIDE)
        vlines!(ax, OMEGA_MAX_eV; color = :gray45, linewidth = 1)
        hlines!(ax, 0; color = :gray65, linewidth = 0.8)
    end

    for (i, z) in enumerate(fig3.Z)
        res = engine[(r, z)]
        for ax in (axL, axR)
            lines!(ax, fig3.E, fig3.λ[(r, z)] .* scale; color = COLORS[i], linewidth = 1.8)
        end
        scatter!(axR, to_eV.(ω_coarse), res.λ_scan .* scale;
                 marker = MARKERS[i], markersize = 6, color = (:white, 0.0),
                 strokecolor = COLORS[i], strokewidth = 1.0)
        for ax in (axL, axR)
            scatter!(ax, [res.ω_star_eV], [res.λ_star * scale];
                     marker = MARKERS[i], markersize = 13, color = COLORS[i],
                     strokecolor = :white, strokewidth = 1.5)
        end
    end

    text!(axR, OMEGA_MAX_eV - 0.1, -0.25 * λ_floor; text = "ħωₘₐₓ", align = (:right, :top),
          fontsize = 11, color = :gray35)
    Label(fig[row + 1, 3], "r = $r Å"; rotation = π / 2, fontsize = 13, color = INK,
          tellheight = false)
end

z_elems = [[LineElement(color = COLORS[i], linewidth = 1.8),
            MarkerElement(marker = MARKERS[i], color = COLORS[i], markersize = 11,
                          strokecolor = :white, strokewidth = 1.2)] for i in eachindex(fig3.Z)]
how_elems = [LineElement(color = INK, linewidth = 1.8),
             MarkerElement(marker = :circle, color = (:white, 0.0), strokecolor = INK,
                           strokewidth = 1.0, markersize = 7),
             MarkerElement(marker = :circle, color = INK, markersize = 11),
             PolyElement(color = OUTSIDE)]
Legend(fig[1, 1:2], [z_elems, how_elems],
       [["$z" for z in fig3.Z],
        ["Fig. 3 stored Λ", "engine coarse scan", "engine minimum", "outside window"]],
       ["z (Å)", "   |  "];
       orientation = :horizontal, titleposition = :left, framevisible = false,
       tellwidth = false, labelsize = 12, titlesize = 12, labelcolor = INK, titlecolor = INK,
       patchsize = (22, 10), colgap = 8, groupgap = 4)

colgap!(fig.layout, 1, 14)
colgap!(fig.layout, 2, 4)
rowgap!(fig.layout, 6)

display(fig)

outpath = plotsdir("friction", "NOAu", "negative", "verify_negative_eigenvalue_fig3.pdf")
mkpath(dirname(outpath))
save(outpath, fig)
save(replace(outpath, ".pdf" => ".png"), fig; px_per_unit = 2)
@info "Saved verification figure" outpath
