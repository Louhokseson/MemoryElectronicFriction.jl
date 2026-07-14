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



params_list = dict_list(Dict{String, Any}(
    "impuritymodel" => [:ErpenbeckThossAdsorbate],
    "centre" => [0],
    "position" => [2.9],
    "temperature" => collect(400:-100:100),

    ## extra [] to make collect(...) as a whole a single parameter as a whole collect(0.05:0.01:0.1)
    "energy" => [collect(0.0001:0.01:30.0)],

    "Γ" => [1.0], # if impuritymodel is ErpenbeckThossAdsorbate, then Γ is used to compute V̄ₖ, which is the prefactor of the Lorentzian width Δ = 2π * Vₖ^2
))

# just make sure that params_list is a list with Dicts
if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end

fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
ax = MyAxis(fig[1,1], xlabel="ω / eV", ylabel= "Λ(ω)",limits=(nothing, nothing, nothing, nothing))


for (i,params_dict) in enumerate(params_list)

    adsorbate_m, position_au, energy_au, temperature_au = buildSystemBath(params_dict)
    Λ_au = FrequencyLambda.Lambda.(energy_au, Ref(adsorbate_m), Ref(position_au), Ref(temperature_au))

    γ_au = MarkovianLambda.widebandfriction(adsorbate_m, position_au, temperature_au)

    γ = ones(length(energy_au)) .* γ_au .* auconvert.(u"u", 1) ./ auconvert.(u"ps",1)

    ω_ev = ustrip.(auconvert.(u"eV", energy_au))

    Λ = Λ_au .* auconvert.(u"u", 1) ./ auconvert.(u"ps",1)

    lines!(ax, ω_ev, Λ; color=colormap[i], linewidth=2, label = "T = $(params_dict["temperature"]) K" )
    lines!(ax, ω_ev, γ; color=colormap[i], linestyle=:dash, linewidth=2 )
    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:right, margin=(5, 5, 5, 5), orientation=:vertical)


    Label(fig[1,1], " $(params_list[1]["impuritymodel"]) \n Position = $(params_list[1]["position"]) Å \n Γ = $(params_list[1]["Γ"]) eV"; tellwidth=false, tellheight=false, valign=:center, halign=:right, padding=(10,10,10,10),fontsize=16)
end

fig