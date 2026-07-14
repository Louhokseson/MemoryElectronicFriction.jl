using DrWatson
@quickactivate "MemoryElectronicFriction" ## Activate project
using HDF5
using Statistics
using LaTeXStrings
using CairoMakie
using HokseonPlots
using MemoryElectronicFriction
using Unitful, UnitfulAtomic
using NQCModels.QuantumModels


# ---- Select which precomputed grid to plot (keys match sim_friction_contour.jl) ----
# domega_eV / Nx select the grid *resolution* (folded into the sim filename).
# Set either to `nothing` to ignore it (e.g. when only one resolution exists).
plot_params = Dict{String, Any}(
    "Gamma_eV"  => 0.2,
    "T_K"       => 300.0,
    "domega_eV" => 0.01,
    "Nx"        => 1000,
)
@unpack Gamma_eV, T_K, domega_eV, Nx = plot_params

# au friction (me / au_t) → u·ps⁻¹
const u_per_ps = ustrip(auconvert(u"u", 1)) / ustrip(auconvert(u"ps", 1))


## Read the data saved by sim_friction_contour.jl.
## The full grid (ω, x) is stored inside each .h5, so the dict is just used to find the right file based on the scalars (Γ, T, domega_eV, Nx) that define the grid resolution. This avoids having to hardcode the grid parameters in both places and risking mismatch.
dir = datadir("sims", "friction", "ErpenbeckThoss")
matches = filter(readdir(dir)) do f
    endswith(f, ".h5") && startswith(f, "Lambda_contour") || return false
    g, t, ω, x = h5open(joinpath(dir, f), "r") do fid
        read(HDF5.attributes(fid)["Gamma_eV"]), read(HDF5.attributes(fid)["T_K"]),
        read(fid, "omega_eV"), read(fid, "x_Ang")
    end
    dω = length(ω) > 1 ? ω[2] - ω[1] : 0.0
    g == Gamma_eV && t == T_K &&
        (domega_eV === nothing || isapprox(dω, domega_eV)) &&
        (Nx === nothing || length(x) == Nx)
end
isempty(matches) && error("No grid file for Γ=$(Gamma_eV) eV, T=$(T_K) K, domega_eV=$(domega_eV), Nx=$(Nx) in $dir — run sim_friction_contour.jl first")
length(matches) > 1 && @warn "Multiple grid files match the selection; using the first" matches
full_data_path = joinpath(dir, first(matches))

Λ_matrix_au, ω_eV, x_Å = h5open(full_data_path, "r") do f
    read(f, "Lambda_au"), read(f, "omega_eV"), read(f, "x_Ang")  # [N_ω, N_x], N_ω, N_x
end

# Plot ω on the x-axis and x on the y-axis: Makie wants z with shape
# (length(x_coord), length(y_coord)) = (N_ω, N_x). Λ_matrix_au is already
# [N_ω, N_x], so no transpose is needed.
Λ_matrix = Λ_matrix_au .* u_per_ps   # [N_ω, N_x], u·ps⁻¹



## Color range: only keep finite, positive values for log color scale. Use robust quantiles to avoid outliers dominating the plot.
Λ_plot = map(v -> (isfinite(v) && v > 0) ? v : NaN, Λ_matrix)
kept   = filter(v -> isfinite(v) && v > 0, vec(Λ_plot))
#clim   = (quantile(kept, 0.01), quantile(kept, 0.99))      # robust log color range
clim = extrema(kept)
# Contour levels on integer decades (powers of ten) so each line is a clean 10ⁿ.
# Fall back to 8 log-spaced levels if the data spans less than one full decade.
emin, emax = ceil(Int, log10(clim[1])), floor(Int, log10(clim[2]))
levels = emax >= emin ? [10.0^e for e in emin:emax] :
                        exp10.(range(log10(clim[1]), log10(clim[2]), length = 8))

# Render a level value as a 10ⁿ string using Unicode superscripts (for labels).
const _SUP = Dict('-'=>'⁻','0'=>'⁰','1'=>'¹','2'=>'²','3'=>'³','4'=>'⁴',
                  '5'=>'⁵','6'=>'⁶','7'=>'⁷','8'=>'⁸','9'=>'⁹')
pow10_label(lvl) = "10" * join(_SUP[c] for c in string(round(Int, log10(lvl))))

# NaN-aware box mean over a (2w+1)² window — used to lightly smooth the *contour*
# source (in log space) so single-cell artifacts vanish and lines read cleanly.
# The heatmap keeps the raw data; only contour geometry is smoothed.
function smooth_nan(A; w = 1)
    nx, ny = size(A)
    B = fill(NaN, nx, ny)
    for j in 1:ny, i in 1:nx
        s = 0.0; c = 0
        for dj in -w:w, di in -w:w
            ii, jj = i + di, j + dj
            (1 <= ii <= nx && 1 <= jj <= ny) || continue
            v = A[ii, jj]
            isnan(v) && continue
            s += v; c += 1
        end
        c > 0 && (B[i, j] = s / c)
    end
    return B
end
Λ_contour = exp10.(smooth_nan(log10.(Λ_plot); w = 3))   # smoothed in log space

# ---- Plot: single Λ(ω, x) panel ----
fig = Figure(size = (HokseonPlots.TWO_COLUMN_WIDTH * 1, HokseonPlots.TWO_COLUMN_WIDTH * 1),
             figure_padding = (4, 4, 4, 4),
             fontsize = 18,
             fonts = (; regular = projectdir("fonts", "MinionPro-Capt.otf")))

ax1 = Axis(fig[1, 1], xlabel = L"\hbar\omega\ (\mathrm{eV})", ylabel = L"x\ (\mathrm{\AA})",
           title = L"\Gamma = %$(Gamma_eV)\ \mathrm{eV},\quad T = %$(round(Int, T_K))\ \mathrm{K}",
           titlesize = 18, xlabelsize = 22, ylabelsize = 22,
           xticklabelsize = 16, yticklabelsize = 16,
           # Outward-pointing major + minor ticks.
           xtickalign = 0, ytickalign = 0, xminortickalign = 0, yminortickalign = 0,
           xminorticksvisible = true, yminorticksvisible = true,
           xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5),
           xticksize = 6, yticksize = 6, xminorticksize = 3, yminorticksize = 3,
           limits = (0.0, maximum(ω_eV), minimum(x_Å), maximum(x_Å)))
hm1 = heatmap!(ax1, ω_eV, x_Å, Λ_plot;
               colormap = :viridis, colorscale = log10,
               colorrange = clim, highclip = :magenta, lowclip = :black,
               nan_color = :magenta)   # mask: non-finite or ≤0 (Λ filtered out)
# Major/minor contour scheme: thin lines at every decade for structure; thicker
# *labeled* lines for the chosen decades. The crowded core decades (10³, 10⁵) are
# left as minor lines so the divergence centre doesn't get cluttered with labels.
# Place one clean label for a decade `10^e` on the near-horizontal sweeping
# contours (which Makie would otherwise label more than once). We fix x near the
# frame centre and find the ω where that column crosses the level → mid-frame.
# Find the ω where the `level` contour crosses the column at x = xq (NaN if none).
function cross_ω(level, xq)
    i   = argmin(abs.(x_Å .- xq))
    col = @view Λ_contour[:, i]
    for j in 1:length(col)-1
        a, b = col[j], col[j+1]
        (isnan(a) || isnan(b)) && continue
        if (a - level) * (b - level) <= 0 && a != b
            return ω_eV[j] + (level - a) / (b - a) * (ω_eV[j+1] - ω_eV[j])
        end
    end
    return NaN
end

# `tilt` scales the y-component of the tangent when mapping the data-space slope
# onto screen space; nudge it if the axis box isn't square and the label sits
# slightly off the curve.
function label_at_x!(ax, e, x_t; dω = 0.16, tilt = 1.0)
    level = 10.0^e
    ωc    = cross_ω(level, x_t)
    isnan(ωc) && return
    # Local contour tangent for the label rotation: step a few cells in x, see how
    # the crossing ω shifts, then turn that data displacement into a screen angle
    # so the label runs *along* the (near-vertical) curve.
    dx     = 5 * (x_Å[2] - x_Å[1])
    ωp, ωm = cross_ω(level, x_t + dx), cross_ω(level, x_t - dx)
    rot    = (isnan(ωp) || isnan(ωm)) ? π/2 :
             atan((2dx) / (maximum(x_Å) - minimum(x_Å)) * tilt,
                  (ωp - ωm) / maximum(ω_eV))
    # offset just beside the line (along ω, the x-axis) into clear background
    text!(ax, ωc + dω, x_t; text = pow10_label(level), color = :white,
          fontsize = 16, align = (:center, :center), rotation = rot)
end

# All decades drawn as lines; compact decades auto-labelled (one label each),
# the long sweeping decades (10⁻¹, 10⁰) get a single manual label to avoid the
# duplicate labels Makie places on long contours.
thick_exps    = (-2, -1, 0, 1, 2, 3, 4)
# Minor lines only for the decades that are NOT drawn thick, otherwise a thin
# line of the same level runs through the labelled gap and crosses the number.
minor_levels  = [10.0^e for e in emin:emax if !(e in thick_exps)]
thick_levels  = [10.0^e for e in emin:emax if e in thick_exps]
# Skip the 10⁴ label: its contour is a tiny arc hugging the ω≈0 / x≈2 Å peak, so
# Makie crams the label against the frame where it clips into nonsense. The
# colorbar max already conveys 10⁴, so the line stays (thick) but unlabelled, and
# 10³ becomes the innermost labelled decade instead.
auto_levels   = [10.0^e for e in emin:emax if e in (-2, 1, 2, 3)]
contour!(ax1, ω_eV, x_Å, Λ_contour; color = (:white, 0.45), linewidth = 0.7,
         levels = minor_levels)
contour!(ax1, ω_eV, x_Å, Λ_contour; color = (:white, 0.95), linewidth = 1.4,
         levels = thick_levels)
contour!(ax1, ω_eV, x_Å, Λ_contour; color = (:white, 0.95), linewidth = 1.4,
         levels = auto_levels, labels = true, labelcolor = :white, labelsize = 16,
         labelformatter = pow10_label)
label_at_x!(ax1,  0, 2.02)   # single clean 10⁰ on the diagonal (mid-frame)
label_at_x!(ax1, -1, 1.97)   # single clean 10⁻¹ on the diagonal (mid-frame)
#hlines!(ax1, [0.0]; linestyle = :dash, color = :red, linewidth = 2, label = "Markovian (ω=0)")
# Clean decade ticks on the (log) colorbar, formatted as 10ⁿ via superscripts.
cbar_ticks = [10.0^e for e in emin:emax]
Colorbar(fig[1, 2], hm1, label = L"\mathcal{K}\ (\mathrm{u⋅ps^{-1}})",
         labelsize = 22, ticklabelsize = 16,
         ticks = cbar_ticks, tickformat = vs -> pow10_label.(vs))


# ---- Annotation block ----
adsorbate_ref = ErpenbeckThossAdsorbate(Γ = austrip(Gamma_eV * u"eV"))
h_eV_at = x -> ustrip(MemoryElectronicFriction.adsorbate_h(austrip(x * u"Å"), adsorbate_ref) * auconvert(u"eV", 1))
Δ_eV_at = x -> ustrip(MemoryElectronicFriction.Δ(austrip(x * u"Å"), adsorbate_ref) * auconvert(u"eV", 1))

info_str = "ErpenbeckThossAdsorbate\n" *
           "Γ = $(Gamma_eV) eV,  T = $(T_K) K\n" *
           "h(1.78 Å) = $(round(h_eV_at(1.78), digits=3)) eV   (Morse min)\n" *
           "h(3.50 Å) = $(round(h_eV_at(3.50), digits=3)) eV   (coupling switch x̃)\n" *
           "Δ(1.78 Å) = $(round(Δ_eV_at(1.78), digits=3)) eV\n" *
           "Δ(3.50 Å) = $(round(Δ_eV_at(3.50), digits=3)) eV\n" *
           "Red dashed: Markovian column (ω=0)"

# Floating Γ annotation inside the panel. Crucial: tellwidth/tellheight = false so
# the Label does NOT drive the sizing of the fig[1,1] cell it shares with ax1 —
# with the defaults (both true) the layout collapses the axis down to the label.
Label(
    fig[1, 1],
    L"\Gamma = %$(Gamma_eV)\ \mathrm{eV}",
    halign = :right,
    valign = :bottom,
    tellwidth = false,
    tellheight = false,
    padding = (8, 0, 8, 0),
    color = :white,
)

#Label(fig[2, 1:2], info_str;
#      tellwidth = false, tellheight = true,
#      halign = :center, padding = (10, 10, 10, 10), fontsize = 14)

display(fig)

save(plotsdir("friction", "ErpenbeckThoss", "single_contour_Γ=$(Gamma_eV).pdf"), fig)