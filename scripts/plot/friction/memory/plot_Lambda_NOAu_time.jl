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

# Numerical cosine (half-range) Fourier transform via the trapezoidal rule:
#   K(t) = (2/π) ∫₀^∞ Λ(ω) cos(ωt) dω
# Inputs:  ω in au (Hartree),  Λ in au,  t_arr in au_time
# Output:  K(t) in au (me/au_time²)
function cosine_transform(ω_au::Vector{Float64}, Λ::Vector{Float64},
                          t_arr::AbstractVector{Float64})
    Δω = diff(ω_au)
    map(t_arr) do t
        integrand = Λ .* cos.(ω_au .* t)
        (2/π) * sum(0.5 .* (integrand[1:end-1] .+ integrand[2:end]) .* Δω)
    end
end


params_list = dict_list(Dict{String, Any}(
    "r"           => [3.6],                          # N–O bond length / Å
    "z"           => [1.6],                          # molecule–surface distance / Å
    "temperature" => [500],

    ## extra [] so collect(...) is treated as a single parameter
    "energy"      => [collect(0.01:0.001:10.0)],      # ω / eV  (avoid 0 — Lambda divides by ω)
))

if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end

# Derive the time axis in atomic units from the ω grid (same for all params since "energy" is fixed)
# au_ps: 1 au_time in ps  →  t [ps] = t_au × au_ps
const au_ps        = ustrip(auconvert(u"ps", 1))                        # ≈ 2.4188e-5 ps/au_time
const unit_conv_K  = ustrip(auconvert(u"u", 1)) / au_ps^2               # au (me/au_t²) → u⋅ps⁻²

_energy_au_ref = austrip.(params_list[1]["energy"] .* u"eV")
_t_max_au      = π / (_energy_au_ref[2] - _energy_au_ref[1])            # Nyquist upper limit
_t_max_au      = 0.2 / au_ps                                            # cap at 0.2 ps equivalent
const t_au     = collect(range(-0.0, _t_max_au, length=3000))
const t_ps     = t_au .* au_ps                                          # for plotting axes [ps]

xmax = 0.1

fig = Figure(size=(HokseonPlots.RESOLUTION[1]*3, 6*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))

ax_rr    = MyAxis(fig[1,1], ylabel="Krr(t) / u⋅ps⁻²",                          limits=(0.0, xmax,    nothing, nothing))
ax_zz    = MyAxis(fig[2,1], ylabel="Kzz(t) / u⋅ps⁻²",                          limits=(0.0, xmax,    nothing, nothing))
ax_rz    = MyAxis(fig[3,1], ylabel="Krz(t) / u⋅ps⁻²",                          limits=(0.0, xmax,    nothing, nothing))
ax_λmin  = MyAxis(fig[4,1], ylabel="λₘᵢₙ(t) / u⋅ps⁻²",                         limits=(0.0, xmax,    nothing, nothing))
ax_λmax  = MyAxis(fig[5,1], xlabel="t / ps", ylabel="λₘₐₓ(t) / u⋅ps⁻²",        limits=(0.0, xmax,    nothing, nothing))

ax_∫λmin  = MyAxis(fig[4,2], ylabel="∫λₘᵢₙ dt / u⋅ps⁻¹", yaxisposition=:right, limits=(0.0, nothing, nothing, nothing))
ax_∫λmax  = MyAxis(fig[5,2], xlabel="t / ps", ylabel="∫λₘₐₓ dt / u⋅ps⁻¹", yaxisposition=:right, limits=(0.0, nothing, nothing, nothing))

# Link t axes within each column independently (different x limits per column)
linkxaxes!(ax_rr, ax_zz, ax_rz, ax_λmin, ax_λmax)   # col 1: capped at xmax
linkxaxes!(ax_∫λmin, ax_∫λmax)                        # col 2: auto-scale

# Hide redundant x-axis decorations — only the bottom row of each column shows ticks/label
hidexdecorations!(ax_rr;    grid=false)
hidexdecorations!(ax_zz;    grid=false)
hidexdecorations!(ax_rz;    grid=false)
hidexdecorations!(ax_λmin;  grid=false)
hidexdecorations!(ax_∫λmin; grid=false)
# ax_λmax (fig[5,1]) and ax_∫λmax (fig[5,2]) are the bottom rows — keep their x decorations

linestyles = [:solid, :solid, :dashdotdot]
colors     = [:blue, :green, :red]

for (i, params_dict) in enumerate(params_list)

    # --- frequency-dependent friction → cosine FT → time-domain memory kernel ---
    # K(t) = (2/π) ∫₀^∞ Λ(ω) cos(ωt) dω   computed in au, converted to u⋅ps⁻² for plotting
    # Markovian limit: K_M(t) = γ·δ(t)  →  a delta function, not shown here
    local adsorbate_m, configuration_au, energy_au, temperature_au = buildSystemBath(params_dict)

    # Vector{Matrix{Float64}} — one 2×2 friction matrix per ω
    Λ_au = FrequencyLambda.Lambda(energy_au, adsorbate_m, configuration_au, temperature_au)

    Λ_rr   = [m[1, 1] for m in Λ_au]   # r–r component  [au]
    Λ_zz   = [m[2, 2] for m in Λ_au]   # z–z component  [au]
    Λ_rz   = [m[1, 2] for m in Λ_au]   # off-diagonal   [au]

    K_rr   = cosine_transform(energy_au, Λ_rr, t_au) .* unit_conv_K   # [u⋅ps⁻²]
    K_zz   = cosine_transform(energy_au, Λ_zz, t_au) .* unit_conv_K
    K_rz   = cosine_transform(energy_au, Λ_rz, t_au) .* unit_conv_K

    # Eigenvalues of K(t) at each t — correct order: transform first, then diagonalise
    K_λmin = [eigmin([K_rr[j] K_rz[j]; K_rz[j] K_zz[j]]) for j in eachindex(t_au)]
    K_λmax = [eigmax([K_rr[j] K_rz[j]; K_rz[j] K_zz[j]]) for j in eachindex(t_au)]

    # Running time integral: γ(t) = ∫₀ᵗ K(τ) dτ  [u⋅ps⁻¹]
    # Converges to the Markovian friction as t → ∞.
    # PSD check: γ_λmin(∞) ≥ 0  ↔  ∫₀^∞ λₘᵢₙ(τ) dτ ≥ 0
    Δt      = diff(t_ps)
    γ_λmin  = cumsum(0.5 .* (K_λmin[1:end-1] .+ K_λmin[2:end]) .* Δt)   # [u⋅ps⁻¹]
    γ_λmax  = cumsum(0.5 .* (K_λmax[1:end-1] .+ K_λmax[2:end]) .* Δt)

    T = params_dict["temperature"]
    @info "T = $T K  →  ∫λₘᵢₙ dt = $(round(γ_λmin[end], sigdigits=4)) u⋅ps⁻¹   ∫λₘₐₓ dt = $(round(γ_λmax[end], sigdigits=4)) u⋅ps⁻¹"

    label = "T = $(T) K"
    lines!(ax_rr,   t_ps,        K_rr;   color=colors[i], linestyle=linestyles[i], linewidth=2, label=label)
    lines!(ax_zz,   t_ps,        K_zz;   color=colors[i], linestyle=linestyles[i], linewidth=2)
    lines!(ax_rz,   t_ps,        K_rz;   color=colors[i], linestyle=linestyles[i], linewidth=2)
    lines!(ax_λmin, t_ps,        K_λmin; color=colors[i], linestyle=linestyles[i], linewidth=2)
    lines!(ax_λmax, t_ps,        K_λmax; color=colors[i], linestyle=linestyles[i], linewidth=2)
    lines!(ax_∫λmin, t_ps[2:end], γ_λmin; color=colors[i], linestyle=linestyles[i], linewidth=2)
    lines!(ax_∫λmax, t_ps[2:end], γ_λmax; color=colors[i], linestyle=linestyles[i], linewidth=2)
end

# Zero-reference line: γ < 0 would violate PSD
hlines!(ax_∫λmin, [0.0]; color=:black, linewidth=1, linestyle=:dash)
hlines!(ax_∫λmax, [0.0]; color=:black, linewidth=1, linestyle=:dash)

# --- annotate with model info ---
adsorbate_m, configuration_au, energy_au, temperature_au = buildSystemBath(params_list[1])

h_au  = HokseonReproduce.adsorbate_h(configuration_au, adsorbate_m)
h_eV  = ustrip(h_au * auconvert(u"eV", 1))
Δ_au  = HokseonReproduce.Δ(configuration_au, adsorbate_m)
Δ_eV  = ustrip(Δ_au * auconvert(u"eV", 1))

info_str = "NOAuAdsorbate\nr = $(params_list[1]["r"]) Å  z = $(params_list[1]["z"]) Å\nh = $(round(h_eV, digits=4)) eV\nΔ = $(round(Δ_eV, digits=4)) eV\nK(t) = (2/π) ∫ Λ(ω) cos(ωt) dω\nλₘᵢₙ: eigmin(K)\nλₘₐₓ: eigmax(K)"
Label(fig[2,2], info_str; tellwidth=false, tellheight=false, valign=:bottom, halign=:center, padding=(10,10,10,10), fontsize=16)

Legend(fig[1,2], ax_rr; tellwidth=false, tellheight=false, valign=:bottom, halign=:center, margin=(5,5,5,5))

display(fig)
