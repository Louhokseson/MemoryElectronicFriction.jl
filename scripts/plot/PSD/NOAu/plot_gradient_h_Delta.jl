using Distributed
using DrWatson
@quickactivate "HokseonReproduce" ## Activate project everywhere
import Pkg; Pkg.precompile() ## Precompile packages in master to speed up workers' precompilation
using HDF5
using DelimitedFiles
using HokseonAssistant
using CairoMakie
using LinearAlgebra: eigmin, eigmax, tr, norm, dot
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


function hΔgradient(params_dict::Dict{String, Any})
    @unpack r, z = params_dict

    adsorbate_m      = NOAuAdsorbate()
    configuration_au = SA[austrip(r * u"Å"), austrip(z * u"Å")]  # SA[r, z]

    dhdx_vec = HokseonReproduce.dh_dx(configuration_au, adsorbate_m)
    dΔdx_vec = HokseonReproduce.dΔ_dx(configuration_au, adsorbate_m)

    # The (r, z) gradients ∇h = (∂h/∂r, ∂h/∂z) and ∇Δ = (∂Δ/∂r, ∂Δ/∂z).
    return dhdx_vec, dΔdx_vec
end

function parallelism_score(a, b)
    na, nb = norm(a), norm(b)
    (na == 0 || nb == 0) && return NaN
    c = dot(a, b) / (na * nb)
    return abs(clamp(c, -1, 1))   # clamp guards against float drift past ±1
end

const R_VALUES = collect(1.17:0.01:2.0)   # N–O bond length r / Å (heatmap x-axis)
const Z_VALUES = collect(1.6:0.01:5.0)     # molecule–surface distance z / Å (heatmap y-axis)


# ---- Evaluate |cos∠(∇h, ∇Δ)| over the (r, z) grid ----
# score[i, j] ↔ (r = R_VALUES[i], z = Z_VALUES[j]); shape (N_r, N_z) is exactly
# what Makie's heatmap!(ax, xs, ys, zmatrix) / contour! expect (no transpose).
score = Matrix{Float64}(undef, length(R_VALUES), length(Z_VALUES))
for (i, r) in enumerate(R_VALUES), (j, z) in enumerate(Z_VALUES)
    params_dict  = Dict{String, Any}("r" => r, "z" => z)
    dhdx, dΔdx   = hΔgradient(params_dict)
    score[i, j]  = parallelism_score(dhdx, dΔdx)
end


# ---- Plot: |cos∠(∇h, ∇Δ)| heatmap with contour overlay ----
fig = Figure(size = (HokseonPlots.TWO_COLUMN_WIDTH, HokseonPlots.TWO_COLUMN_WIDTH),
             figure_padding = (4, 4, 4, 4),
             fontsize = 18,
             fonts = (; regular = projectdir("fonts", "MinionPro-Capt.otf")))

ax = Axis(fig[1, 1];
          xlabel = L"r\ (\mathrm{\AA})", ylabel = L"z\ (\mathrm{\AA})",
          title  = "NO/Au(111) gradient parallelism score",
          titlefont = projectdir("fonts", "MinionPro-Capt.otf"),
          titlesize = 18, xlabelsize = 22, ylabelsize = 22,
          xticklabelsize = 16, yticklabelsize = 16,
          xtickalign = 0, ytickalign = 0, xminortickalign = 0, yminortickalign = 0,
          xminorticksvisible = true, yminorticksvisible = true,
          xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5),
          xticksize = 6, yticksize = 6, xminorticksize = 3, yminorticksize = 3,
          limits = (extrema(R_VALUES)..., extrema(Z_VALUES)...))

hm = heatmap!(ax, R_VALUES, Z_VALUES, score;
              colormap = :viridis, colorrange = (0, 1), nan_color = :gray)
contour!(ax, R_VALUES, Z_VALUES, score;
         color = (:white, 0.6), linewidth = 1.0, levels = 0.1:0.1:0.9)

Colorbar(fig[1, 2], hm; label = L"|\cos\angle(\nabla h,\ \nabla\Delta)|",
         labelsize = 22, ticklabelsize = 16, ticks = 0:0.2:1)

display(fig)

mkpath(plotsdir("PSD", "NOAu"))
save(plotsdir("PSD", "NOAu", "gradient_parallelism_h_Delta.pdf"), fig)