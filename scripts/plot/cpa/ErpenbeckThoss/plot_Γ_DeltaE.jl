using Distributed
using DrWatson
@quickactivate "HokseonReproduce"

using Unitful, UnitfulAtomic
using LinearAlgebra: eigen
using StaticArrays: SA
using Random
using HDF5
using HokseonAssistant
using HokseonPlots
using CairoMakie
using ColorSchemes
using Colors

# Sequential temperature colour ramp from HokseonPlots' NICECOLORS — picks N
# evenly spaced colours so cool→warm reads as low-T→high-T.
function temperature_palette(Ts)
    n = length(Ts)
    n == 1 ? [HokseonPlots.NICECOLORS[end-1]] :
             [HokseonPlots.NICECOLORS[Int(round(1 + (length(HokseonPlots.NICECOLORS)-1) * (i-1)/(n-1)))]
              for i in 1:n]
end

HokseonAssistant.julia_build_procs()
@everywhere using HokseonReproduce
@everywhere using StaticArrays: SA
@everywhere using Unitful, UnitfulAtomic

# ---------------------------------------------------------------------------
# Variant detection mirrors CPA_dict_to_data_savename: memory CPA carries
# `kernel_average`, Markovian does not.
# ---------------------------------------------------------------------------
variant_of(cfg) = haskey(cfg, "kernel_average") ? :memory : :markovian
variant_label(::Val{:memory})    = "memory CPA"
variant_label(::Val{:markovian}) = "Markovian CPA"

# ---------------------------------------------------------------------------
# Read one (params_list, cfg) sweep into (Γ, ΔE) vectors. Skips and warns on
# missing files so a partially-finished sweep still plots cleanly.
# ---------------------------------------------------------------------------
function load_Γ_DeltaE(params_list, cfg)
    Γ_eV      = Float64[]
    DeltaE_eV = Float64[]
    for p in params_list
        savingpath, savingname = CPA_dict_to_data_savename(p, cfg)
        full_data_path = datadir(savingpath, savingname)
        if !isfile(full_data_path)
            @warn "Missing CPA data file" full_data_path; continue
        end
        ΔE_au = h5open(full_data_path, "r") do f
            read(f, "DeltaE_au")
        end
        push!(Γ_eV,      ustrip(u"eV", p["Γ"]))
        push!(DeltaE_eV, ustrip(auconvert(u"eV", first(ΔE_au))))
    end
    order = sortperm(Γ_eV)
    return Γ_eV[order], DeltaE_eV[order]
end

# ---------------------------------------------------------------------------
# Publication-quality ΔE-vs-Γ comparison plot.
#
# `configs` is a Vector of CPA_config dicts (mixed memory/Markovian, mixed
# T_K). Curves are colour-coded by `T_K` (cool→warm) and styled by variant
# (memory: solid + filled circle; Markovian: dashed + open circle). A
# split legend factorises temperature and variant so each visual encoding
# is decoded once by the reader.
# ---------------------------------------------------------------------------
function plot_Γ_DeltaE(params_list, configs::AbstractVector;
                       title       = nothing,
                       xscale      = log10,
                       xticks      = nothing,
                       figsize     = (HokseonPlots.RESOLUTION[1] * 2.0,
                                      HokseonPlots.RESOLUTION[2] * 3.0),
                       legend_pos  = (:top, :left))

    Ts        = sort(unique(c["T_K"] for c in configs))
    color_map = Dict(Ts .=> temperature_palette(Ts))
    variants  = unique(variant_of.(configs))

    fig = Figure(; size = figsize,
                   figure_padding = (8, 12, 6, 6),
                   fonts = (; regular = projectdir("fonts", "MinionPro-Capt.otf")))
    axis_kwargs = (
        xlabel = "Δ / eV",
        ylabel = "ΔE / eV",
        xscale = xscale,
        # Pin major x-ticks to the swept Γ values so every plotted point
        # sits on a labelled tick. Minor ticks suppressed on x because the
        # majors already cover all data; minor ticks left on y.
        xminorticksvisible = xticks === nothing,
        yminorticksvisible = true,
        xminorgridvisible  = false,
        yminorgridvisible  = false,
    )
    ax = xticks === nothing ?
        MyAxis(fig[1, 1]; axis_kwargs...) :
        MyAxis(fig[1, 1]; axis_kwargs..., xticks = xticks)

    for cfg in configs
        Γ_eV, ΔE_eV = load_Γ_DeltaE(params_list, cfg)
        isempty(Γ_eV) && continue
        T   = cfg["T_K"]
        col = color_map[T]
        v   = variant_of(cfg)
        if v == :memory
            lines!(ax, Γ_eV, ΔE_eV;
                   color = col, linewidth = 1.8, linestyle = :solid)
            scatter!(ax, Γ_eV, ΔE_eV;
                     color = col, marker = :circle, markersize = 9,
                     strokecolor = col, strokewidth = 0.6)
        else
            lines!(ax, Γ_eV, ΔE_eV;
                   color = col, linewidth = 1.8, linestyle = :dash)
            scatter!(ax, Γ_eV, ΔE_eV;
                     color = :white, marker = :circle, markersize = 9,
                     strokecolor = col, strokewidth = 1.5)
        end
    end

    # Two-block legend: temperature (color) × variant (linestyle/marker).
    T_elems = [[LineElement(; color = color_map[T], linewidth = 1.8),
                MarkerElement(; color = color_map[T], marker = :circle,
                                markersize = 9, strokecolor = color_map[T],
                                strokewidth = 0.6)] for T in Ts]
    T_labels = ["$(T) K" for T in Ts]

    variant_elems = Any[]
    variant_labels = String[]
    if :memory in variants
        push!(variant_elems,
              [LineElement(; color = :black, linewidth = 1.8, linestyle = :solid),
               MarkerElement(; color = :black, marker = :circle, markersize = 9,
                               strokecolor = :black, strokewidth = 0.6)])
        push!(variant_labels, "memory")
    end
    if :markovian in variants
        push!(variant_elems,
              [LineElement(; color = :black, linewidth = 1.8, linestyle = :dash),
               MarkerElement(; color = :white, marker = :circle, markersize = 9,
                               strokecolor = :black, strokewidth = 1.5)])
        push!(variant_labels, "Markovian")
    end

    Legend(fig[1, 1],
           [T_elems, variant_elems],
           [T_labels, variant_labels],
           ["temperature", "CPA"];
           tellwidth   = false, tellheight = false,
           valign      = legend_pos[1],
           halign      = legend_pos[2],
           margin      = (8, 8, 8, 8),
           framevisible = true,
           framecolor   = (:black, 0.4),
           rowgap       = 2,
           titlegap     = 4,
           groupgap     = 10,
           patchsize    = (18, 12),
           labelsize    = 11,
           titlesize    = 12)

    if title !== nothing
        Label(fig[1, 1], title;
              tellwidth = false, tellheight = false,
              valign = :center, halign = :left,
              padding = (0, 8, 6, 0), fontsize = 13)
    end

    return fig
end

# ===========================================================================
# Erpenbeck–Thoss sweep at fixed Eₜ, three temperatures
# ===========================================================================

const E_TRANS_eV = 0.5   # incident translational energy (eV) we compare across
const Γ_eV_LIST  = [0.02, 0.1, 0.25, 0.5, 1.0]   # also used as x-tick locations

all_params_et = Dict{String, Any}(
    "mass"                  => [10.54u"u"],
    "Γ"                     => Γ_eV_LIST .* u"eV",
    "r0"                    => [[5.0u"Å"]],
    "translational_kinetic" => [E_TRANS_eV * u"eV"],
    "state"                 => [1],
    "tmax"                  => [200.0u"fs"],
    "dt"                    => [0.01u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [1],
    "termination_threshold" => [5.0u"Å"],
)
params_list_et = dict_list(all_params_et)

# Three temperatures with both variants → 6 curves.
const T_K_LIST = [10, 300, 1000, 2000]

memory_configs = [Dict{String,Any}(
        "model"          => :ErpenbeckThoss,
        "T_K"            => T,
        "ω"              => collect(0.01:0.01:20.0),
        "stride"         => 1,
        "parallel"       => nworkers() > 1,
        "kernel_average" => :arithmetic,
    ) for T in T_K_LIST]

markovian_configs = [Dict{String,Any}(
        "model"   => :ErpenbeckThoss,
        "T_K"     => T,
        "stride"  => 1,
        "parallel" => false,
    ) for T in T_K_LIST]

configs_et = vcat(memory_configs, markovian_configs)

fig = plot_Γ_DeltaE(params_list_et, configs_et;
                    title  = "ErpenbeckThoss \n Eₜ = $(E_TRANS_eV) eV",
                    xticks = (Γ_eV_LIST, string.(Γ_eV_LIST)))
display(fig)

# Optional save — uncomment to render to disk.
# save_figure(plotsdir("cpa", "ErpenbeckThoss", "DeltaE_vs_Gamma_T_variant"), fig)
