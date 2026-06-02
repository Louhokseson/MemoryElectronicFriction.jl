using DrWatson
@quickactivate "HokseonReproduce" ## Activate project
using HDF5
using Statistics
using LaTeXStrings
using CairoMakie
using HokseonPlots
using HokseonReproduce
using Unitful, UnitfulAtomic
using NQCModels.QuantumModels


# ---- Three Γ panels in a row, attached, sharing one log color scale ----
# Companion to plot_single_contour.jl: same Λ(ω, x) map, but compared across
# Γ = 0.02, 0.2, 0.8 eV at fixed T. The panels share the colour range and contour
# levels so the divergence strength can be read across Γ at a glance.
Γ_list    = [0.02, 0.2, 0.8]      # eV, left → right
T_K       = 300.0
domega_eV = 0.01
Nx        = 1000

# au friction (me / au_t) → u·ps⁻¹
const u_per_ps = ustrip(auconvert(u"u", 1)) / ustrip(auconvert(u"ps", 1))

dir = datadir("sims", "friction", "ErpenbeckThoss")

# Locate the grid file for a given Γ (the grid resolution is encoded by the
# scalars T, domega_eV, Nx — same matching logic as plot_single_contour.jl).
function find_grid(Gamma_eV)
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
    isempty(matches) && error("No grid file for Γ=$(Gamma_eV) eV, T=$(T_K) K, domega_eV=$(domega_eV), Nx=$(Nx) in $dir")
    length(matches) > 1 && @warn "Multiple grid files match Γ=$(Gamma_eV); using the first" matches
    return joinpath(dir, first(matches))
end

# NaN-aware box mean over a (2w+1)² window — lightly smooths the *contour* source
# (in log space) so single-cell artifacts vanish and lines read cleanly. The
# heatmap keeps the raw data; only contour geometry is smoothed.
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

# Load + prepare one panel's grid: heatmap source (Λ_plot) and smoothed contour
# source (Λ_contour), both [N_ω, N_x] so ω is the x-axis and x the y-axis.
function load_panel(Gamma_eV)
    Λ_au, ω_eV, x_Å = h5open(find_grid(Gamma_eV), "r") do f
        read(f, "Lambda_au"), read(f, "omega_eV"), read(f, "x_Ang")  # [N_ω, N_x]
    end
    Λ_plot    = map(v -> (isfinite(v) && v > 0) ? v * u_per_ps : NaN, Λ_au)
    Λ_contour = exp10.(smooth_nan(log10.(Λ_plot); w = 3))
    return (; Λ_plot, Λ_contour, ω_eV, x_Å)
end

panels = map(load_panel, Γ_list)

# Shared log colour range + decade contour levels across all panels.
kept = filter(v -> isfinite(v) && v > 0, vcat((vec(p.Λ_plot) for p in panels)...))
clim = extrema(kept)
emin, emax = ceil(Int, log10(clim[1])), floor(Int, log10(clim[2]))

# Shared data limits (panels are linked, so set once). ω is capped at 3.6 eV —
# below the integer ticks (0,1,2,3) and the data max — so each panel ends between
# ticks and the "0" tick can show on every panel without colliding at the seams.
ωhi  = 3.6
xmin = minimum(minimum(p.x_Å) for p in panels)
xmax = maximum(maximum(p.x_Å) for p in panels)

# Render a level value as a 10ⁿ string using Unicode superscripts (for labels).
# Non-const so re-running the script in the same session doesn't warn.
_SUP = Dict('-'=>'⁻','0'=>'⁰','1'=>'¹','2'=>'²','3'=>'³','4'=>'⁴',
            '5'=>'⁵','6'=>'⁶','7'=>'⁷','8'=>'⁸','9'=>'⁹')
pow10_label(lvl) = "10" * join(_SUP[c] for c in string(round(Int, log10(lvl))))

# Contour scheme (mirrors the single-panel plot): thin minor lines at every
# decade for structure, thick lines on the chosen decades, and auto-labelled
# lines on a clean subset. 10⁴ stays an (unlabelled) thick line — its contour is
# a tiny arc at the peak and the colorbar max already conveys it.
thick_exps   = (-2, -1, 0, 1, 2, 3, 4)
# Auto-label these decades. The high decades (≥ 10³) stay unlabelled — they are
# tiny arcs hugging the Γ=0.02 panel's ω≈0 peak and only cram the frame edge.
label_exps   = (-2, -1, 0, 1, 2)
minor_levels = [10.0^e for e in emin:emax if !(e in thick_exps)]
thick_levels = [10.0^e for e in emin:emax if e in thick_exps]
label_levels = [10.0^e for e in emin:emax if e in label_exps]

function draw_panel!(ax, p)
    hm = heatmap!(ax, p.ω_eV, p.x_Å, p.Λ_plot;
                  colormap = :viridis, colorscale = log10,
                  colorrange = clim, highclip = :magenta, lowclip = :black,
                  nan_color = :magenta)
    contour!(ax, p.ω_eV, p.x_Å, p.Λ_contour; color = (:white, 0.45), linewidth = 0.6,
             levels = minor_levels)
    contour!(ax, p.ω_eV, p.x_Å, p.Λ_contour; color = (:white, 0.95), linewidth = 1.2,
             levels = thick_levels)
    contour!(ax, p.ω_eV, p.x_Å, p.Λ_contour; color = (:white, 0.95), linewidth = 1.2,
             levels = label_levels, labels = true, labelcolor = :white,
             labelsize = 13, labelformatter = pow10_label)
    return hm
end

# ---- Figure: 3 attached panels + shared colorbar ----
fig = Figure(size = (HokseonPlots.TWO_COLUMN_WIDTH * 2, HokseonPlots.TWO_COLUMN_WIDTH * 0.8),
             figure_padding = (4, 4, 4, 4),
             fontsize = 15,
             fonts = (; regular = projectdir("fonts", "MinionPro-Capt.otf")))

axes = Axis[]
hms  = Any[]
for (k, (g, p)) in enumerate(zip(Γ_list, panels))
    ax = Axis(fig[1, k];
              title = L"\Delta(x=0) = %$(g/2)\ \mathrm{eV}", titlesize = 16,
              xlabelsize = 20, ylabelsize = 20,
              xticklabelsize = 14, yticklabelsize = 14,
              xtickalign = 0, ytickalign = 0, xminortickalign = 0, yminortickalign = 0,
              xminorticksvisible = true, yminorticksvisible = true,
              xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5),
              xticks = 0:1:floor(Int, ωhi),
              xticksize = 5, yticksize = 5, xminorticksize = 3, yminorticksize = 3,
              limits = (0.0, ωhi, xmin, xmax))
    push!(hms, draw_panel!(ax, p))
    # Panel index (a)/(b)/(c) in the bottom-right corner, axis-relative so it
    # sits consistently across panels regardless of the underlying data.
    text!(ax, 0.97, 0.04; text = "($(('a' + k - 1)))", space = :relative,
          align = (:right, :bottom), color = :white, fontsize = 16, font = :bold)
    if k == 1
        ax.ylabel = L"x\ (\mathrm{\AA})"
    else
        hideydecorations!(ax; grid = false)   # only the leftmost panel keeps the y axis
    end
    push!(axes, ax)
end
linkaxes!(axes...)

# Shared colorbar with clean decade ticks formatted as 10ⁿ.
cbar_ticks = [10.0^e for e in emin:emax]
Colorbar(fig[1, 4], first(hms), label = L"\mathcal{K}\ (\mathrm{u⋅ps^{-1}})",
         labelsize = 20, ticklabelsize = 14,
         ticks = cbar_ticks, tickformat = vs -> pow10_label.(vs))

# Attach the three panels (zero gap); keep a small gap before the colorbar.
# (Must run after the Colorbar exists, so column gap 3 is valid.)
colgap!(fig.layout, 1, 0)
colgap!(fig.layout, 2, 0)
colgap!(fig.layout, 3, 8)

# Shared x-label under the panels. (T super-title disabled for now — re-enabling
# the fig[0, 1:3] Label adds a row 0 and shifts every row-gap index up by one.)
Label(fig[2, 1:3], L"\hbar\omega\ (\mathrm{eV})", fontsize = 20, tellwidth = false)
#Label(fig[0, 1:3], L"T = %$(round(Int, T_K))\ \mathrm{K}", fontsize = 16, tellwidth = false)
rowgap!(fig.layout, 1, 2)   # panels → x-label

display(fig)

save(plotsdir("friction", "ErpenbeckThoss", "Gamma_contour.png"), fig)
