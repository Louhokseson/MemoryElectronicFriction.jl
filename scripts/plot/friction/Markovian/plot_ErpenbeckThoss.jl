using DrWatson
@quickactivate "MemoryElectronicFriction"

using MemoryElectronicFriction

# Plotting packages
using CairoMakie
using HokseonPlots
using HokseonAssistant
using ColorSchemes
using NQCModels
using NQCDynamics
using NQCCalculators
using Colors
using Unitful, UnitfulAtomic
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;
HokseonAssistant.julia_session();

params_list = dict_list(Dict{String, Any}(
    "impuritymodel" => [:ErpenbeckThossAdsorbate],
    "centre" => [0],
    "position" => [collect(2.0:0.001:2.05)],
    "temperature" => [300],
    "Γ" => [0.06],
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

function NQCD_MarkovianFriction(params_dict::Dict{String, Any}; nstates, bandwidth)
    @unpack  position, impuritymodel, temperature = params_dict

    temperature_au = austrip.(temperature*u"K")
    position_au = austrip.(position*u"Å")

    if impuritymodel == :ErpenbeckThossAdsorbate
        Γ = params_dict["Γ"]
        adsorbate_m = NQCModels.ErpenbeckThoss(;Γ=austrip(Γ*u"eV"))
        model = WideBandBath(adsorbate_m; step=bandwidth/nstates, bandmin=-bandwidth/2, bandmax=bandwidth/2)
        atoms = Atoms(1u"u")
        sim = Simulation{DiabaticMDEF}(atoms, model, friction_method=NQCCalculators.WideBandExact(model.ρ, 1/temperature_au))

        MarkovianFriction = [only(NQCCalculators.evaluate_friction(sim.cache, hcat(x))) for x in position_au]
        return MarkovianFriction
    end
end

function plot_MarkovianLambda(params_list)
    ## Plotting set up
    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="x / Å", ylabel= "Λ(x) / u⋅ps⁻¹",limits=(2.0, 2.05, -200000, 200000))

    bandwidth = 400
    bandwidth_au = austrip.(bandwidth*u"eV")
    nstates = 1000
    for (i,params_dict) in enumerate(params_list)
        @info "NQCD extrema"
        MarkovianFriction_NQCD_au = NQCD_MarkovianFriction(params_dict; nstates, bandwidth = bandwidth_au)

        MarkovianFriction_NQCD = ustrip.(MarkovianFriction_NQCD_au .* auconvert.(u"u", 1) ./ auconvert.(u"ps",1))

        adsorbate_m, position_au, temperature_au = buildSystemBath(params_dict)

        @info "HokseonRrepoduce A"

        Λ_au = MarkovianLambda.widebandfriction.(Ref(adsorbate_m), position_au, Ref(temperature_au))

        Λ_ps⁻¹ = ustrip.(Λ_au .* auconvert.(u"u", 1) ./ auconvert.(u"ps",1))

        x_Å = ustrip.(auconvert.(u"Å", position_au))

        lines!(ax, x_Å, -Λ_ps⁻¹; color=:red, linewidth=2,
        label = "HoksesonReproduce × -1" )

        lines!(ax, x_Å, MarkovianFriction_NQCD; color=:red, linewidth=2, linestyle=:dash,
        label = "NQCD" )

        lines!(ax, x_Å, MarkovianFriction_NQCD .- Λ_ps⁻¹; color=:black, linewidth=2, linestyle=:dash, label = "Difference" )

        hlines!(ax, [-61639.41800270504]; color=:blue, linewidth=2, label = "-61639 eV" )

    end
    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:right, margin=(5, 5, 5, 5), orientation=:vertical)

    Label(fig[1,1], " $(params_list[1]["impuritymodel"]) \n Γ = $(params_list[1]["Γ"]) eV \n T = $(params_list[1]["temperature"]) K \n NQCD.nstates = $(nstates) \n NQCD.bandwidth = $(bandwidth) eV"; tellwidth=false, tellheight=false, valign=:bottom, halign=:right, padding=(10,10,10,10),fontsize=16)
    return fig

end

display(plot_MarkovianLambda(params_list))