using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded only once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end

# Plotting packages
using CairoMakie
using HokseonPlots
using HokseonAssistant
using ColorSchemes
using NQCModels
using Colors
using Unitful, UnitfulAtomic
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;
HokseonAssistant.julia_session();

params_list = dict_list(Dict{String, Any}(
    "nstates" => [20],
    "width" => [4],
    "discretisation" => [NQCModels.ShenviGaussLegendre],
    "impuritymodel" => [:ErpenbeckThossAdsorbate],
    "centre" => [0],
    "position" => [collect(1.0:0.01:5.0)],
    "temperature" => collect(300:-100:100),
    "Γ" => [0.5],
))

function buildSystemBath(params_dict::Dict{String, Any})
    @unpack  position, impuritymodel, temperature = params_dict

    if impuritymodel == :ErpenbeckThossAdsorbate
        Γ = params_dict["Γ"]
        adsorbate_m = ErpenbeckThossAdsorbate(Γ=austrip(Γ*u"eV"))
    else
        adsorbate_m = eval(impuritymodel)()
    end


    temperature_au = austrip.(temperature*u"K")
    position_au = austrip.(position*u"Å")

    return adsorbate_m, position_au, temperature_au
end

function plot_MarkovianLambda(params_list)
    ## Plotting set up
    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="x / Å", ylabel= "Λ(x)",limits=(nothing, nothing, nothing, nothing))

    for (i,params_dict) in enumerate(params_list)

        adsorbate_m, position_au, temperature_au = buildSystemBath(params_dict)

        Λ_au = MarkovianLambda.widebandfriction.(Ref(adsorbate_m), position_au, Ref(temperature_au))

        Λ_ps⁻¹ = Λ_au .* auconvert.(u"u", 1) ./ auconvert.(u"ps",1)

        x_Å = ustrip.(auconvert.(u"Å", position_au))

        lines!(ax, x_Å, Λ_ps⁻¹; color=colormap[i], linewidth=2,
        label = "T = $(params_dict["temperature"]) K" )

    end
    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:right, margin=(5, 5, 5, 5), orientation=:vertical)

    Label(fig[1,1], " $(params_list[1]["impuritymodel"]) \n Γ = $(params_list[1]["Γ"]) eV"; tellwidth=false, tellheight=false, valign=:center, halign=:right, padding=(10,10,10,10),fontsize=16)
    return fig

end

display(plot_MarkovianLambda(params_list))