## Only for the main process
using Distributed
using DrWatson
@quickactivate "HokseonReproduce" ## Activate project everywhere
import Pkg; Pkg.precompile() ## Precompile packages in master to speed up workers' precompilation
using HDF5
using DelimitedFiles
using HokseonAssistant
using CairoMakie
using LinearAlgebra: eigmin, eigmax
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

function buildSystemBath(params_dict::Dict{String, Any})
    @unpack r, z, temperature, energy = params_dict

    adsorbate_m      = NOAuAdsorbate()
    energy_au        = austrip.(energy * u"eV")
    temperature_au   = austrip.(temperature * u"K")
    configuration_au = SA[austrip(r * u"Å"), austrip(z * u"Å")]  # SA[r, z]

    return adsorbate_m, configuration_au, energy_au, temperature_au
end


params_list = dict_list(Dict{String, Any}(
    "r"           => [1.6],                          # N–O bond length / Å
    "z"           => [1.6],                          # molecule–surface distance / Å
    "temperature" => [2500, 500, 1],

    ## extra [] so collect(...) is treated as a single parameter
    "energy"      => [collect(0.01:0.01:10.0)],      # ω / eV  (avoid 0 — Lambda divides by ω)
))

if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end


fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 4*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))

ax_rr   = MyAxis(fig[1,1], ylabel="Λrr(ω) / u⋅ps⁻¹",       limits=(0.0, nothing, nothing, nothing))
ax_zz   = MyAxis(fig[2,1], ylabel="Λzz(ω) / u⋅ps⁻¹",       limits=(0.0, nothing, nothing, nothing))
ax_rz   = MyAxis(fig[3,1], ylabel="Λrz(ω) / u⋅ps⁻¹",       limits=(0.0, nothing, nothing, nothing))
ax_λmin = MyAxis(fig[4,1], ylabel="λₘᵢₙ(ω) / u⋅ps⁻¹",      limits=(0.0, nothing, nothing, nothing))
ax_λmax = MyAxis(fig[5,1], xlabel="ω / eV", ylabel="λₘₐₓ(ω) / u⋅ps⁻¹", limits=(0.0, nothing, nothing, nothing))

# Link ω axes so zoom/pan is shared across all four panels
linkxaxes!(ax_rr, ax_zz, ax_rz, ax_λmin, ax_λmax)

# Hide redundant x-axis decorations on upper panels
hidexdecorations!(ax_rr;   grid=false)
hidexdecorations!(ax_zz;   grid=false)
hidexdecorations!(ax_rz;   grid=false)

linestyles = [:solid, :solid, :dashdotdot]
colors     = [:blue, :green, :red]

unit_conv = ustrip(auconvert(u"u", 1) / auconvert(u"ps", 1))   # au friction → u⋅ps⁻¹

for (i, params_dict) in enumerate(params_list)

    # --- Markovian friction from POGOModel (NQCD) — horizontal lines ---
    local γ_mat   = buildMarkovianFriction(params_dict) .* unit_conv   # 2×2 in u⋅ps⁻¹
    local γ_rr    = γ_mat[1, 1]
    local γ_zz    = γ_mat[2, 2]
    local γ_rz    = γ_mat[1, 2]
    local γ_λmin  = eigmin(γ_mat)
    local γ_λmax  = eigmax(γ_mat)
    markov_label = i == 1 ? "Markovian (NQCD)" : nothing
    hlines!(ax_rr,   [γ_rr];   color=colors[i], linestyle=:dash, linewidth=2, label=markov_label)
    hlines!(ax_zz,   [γ_zz];   color=colors[i], linestyle=:dash, linewidth=2)
    hlines!(ax_rz,   [γ_rz];   color=colors[i], linestyle=:dash, linewidth=2)
    hlines!(ax_λmin, [γ_λmin]; color=colors[i], linestyle=:dash, linewidth=2)
    hlines!(ax_λmax, [γ_λmax]; color=colors[i], linestyle=:dash, linewidth=2)


    # --- frequency-dependent friction from FrequencyLambda (HokseonReproduce) — curves ---
    local adsorbate_m, configuration_au, energy_au, temperature_au = buildSystemBath(params_dict)

    # Vector{Matrix{Float64}} — one 2×2 friction matrix per ω
    Λ_au = FrequencyLambda.Lambda(energy_au, adsorbate_m, configuration_au, temperature_au)

    ω_eV  = ustrip.(auconvert.(u"eV", energy_au))

    Λ_rr  = [m[1, 1] for m in Λ_au] .* unit_conv   # r–r component
    Λ_zz  = [m[2, 2] for m in Λ_au] .* unit_conv   # z–z component
    Λ_rz  = [m[1, 2] for m in Λ_au] .* unit_conv   # off-diagonal r–z

    Λ_λmin = [eigmin(m) for m in Λ_au] .* unit_conv   # smallest eigenvalue per ω
    Λ_λmax = [eigmax(m) for m in Λ_au] .* unit_conv   # largest eigenvalue per ω

    label = "T = $(params_dict["temperature"]) K"
    lines!(ax_rr,   ω_eV, Λ_rr;   color=colors[i], linestyle=linestyles[i], linewidth=2, label=label)
    lines!(ax_zz,   ω_eV, Λ_zz;   color=colors[i], linestyle=linestyles[i], linewidth=2, label=label)
    lines!(ax_rz,   ω_eV, Λ_rz;   color=colors[i], linestyle=linestyles[i], linewidth=2, label=label)
    lines!(ax_λmin, ω_eV, Λ_λmin; color=colors[i], linestyle=linestyles[i], linewidth=2, label=label)
    lines!(ax_λmax, ω_eV, Λ_λmax; color=colors[i], linestyle=linestyles[i], linewidth=2, label=label)
end

# --- annotate with model info ---
adsorbate_m, configuration_au, energy_au, temperature_au = buildSystemBath(params_list[1])

h_au  = HokseonReproduce.adsorbate_h(configuration_au, adsorbate_m)
h_eV  = ustrip(h_au * auconvert(u"eV", 1))
Δ_au  = HokseonReproduce.Δ(configuration_au, adsorbate_m)
Δ_eV  = ustrip(Δ_au * auconvert(u"eV", 1))

vlines!(ax_rr,   abs(h_eV); linestyle=:dash, color=:black, linewidth=2, label="h(r,z)")
vlines!(ax_zz,   abs(h_eV); linestyle=:dash, color=:black, linewidth=2)
vlines!(ax_rz,   abs(h_eV); linestyle=:dash, color=:black, linewidth=2)
vlines!(ax_λmin, abs(h_eV); linestyle=:dash, color=:black, linewidth=2)

info_str = "NOAuAdsorbate\nr = $(params_list[1]["r"]) Å  z = $(params_list[1]["z"]) Å\nh = $(round(h_eV, digits=4)) eV\nΔ = $(round(Δ_eV, digits=4)) eV\nλₘᵢₙ: eigmin(Λ)\nλₘₐₓ: eigmax(Λ)"
Label(fig[2,2], info_str; tellwidth=false, tellheight=false, valign=:bottom, halign=:center, padding=(10,10,10,10), fontsize=16)

Legend(fig[4,2], ax_rr; tellwidth=false, tellheight=false, valign=:bottom, halign=:center, margin=(5,5,5,5))

display(fig)
