using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end

using CairoMakie
using HokseonPlots
using HokseonAssistant
using ColorSchemes
using Colors
using Unitful, UnitfulAtomic
using DelimitedFiles
using LaTeXStrings, Printf
using NQCModels
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;
HokseonAssistant.julia_session()



position = 0.3
position_au = austrip.(position*u"Å")
energies = collect(0.05:0.001:0.5)
energies_au = austrip.(energies*u"eV")
temperatures = collect(5500:-500:4000)
temperatures_au = austrip.(temperatures*u"K")
adsorbate_m = AndersonImpurityModels.BrandbygeAdsorbate()

all_params = Dict{String, Any}(
    "nstates" => [5],
    "width" => [6],
    "temperature" => [300.0],
    "discretisation" => [:GapGaussLegendre],
    "impuritymodel" => :Hokseon,
    "gap" => [0.49],
    "centre" => [0],
    "position" => [1.0],
)

params_list = dict_list(all_params)
# just make sure that params_list is a list with Dicts
if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end

@unpack nstates, width, centre, position = params_list[1]
bandmin = - austrip(((width / 2) - centre) * u"eV")
bandmax = austrip(((width / 2) + centre)* u"eV")
bath = NQCModels.TrapezoidalRule(nstates, bandmin, bandmax)

function Lambda_threaded(energy_vec_au, bath, adsorbate_model, adsorbate_position_au, temperature_au)

    Lambda_au_vec = Vector{Float64}(undef, length(energy_vec_au))

    # Parallelize the loop using @threads
    Threads.@threads for i in eachindex(energy_vec_au)
        Lambda_au_vec[i] = Lambda(
            energy_vec_au[i],
            bath,
            adsorbate_model,
            adsorbate_position_au,
            temperature_au
        )
    end

    return Lambda_au_vec
end




function sim_Lambda(energy_vec_au, bath, adsorbate_model, adsorbate_position_au, temperature_au)
    # Calculate Lambda_au for all energies in a single pass.

    Lambda_au_vec = FrequencyLambda.Lambda(energy_vec_au, bath, adsorbate_model, adsorbate_position_au, temperature_au)
    # Use broadcasting to convert the results to the desired units.
    Lambda_nau_vec = ustrip.(auconvert.(u"fs^-2", Lambda_au_vec))

    return Lambda_nau_vec, Lambda_au_vec
end

function sim_Lambda_batch_single_vector(energy_vec_au, adsorbate_model, adsorbate_position_vec_au, temperature_vec_au)
    
    # Check if temperature is a vector and position is a scalar
    if typeof(temperature_vec_au) <: AbstractVector && !(typeof(adsorbate_position_vec_au) <: AbstractVector)
        
        scalar_value = ustrip(auconvert(u"Å",adsorbate_position_vec_au))
        results = []
        for temp_au in temperature_vec_au
            Lambda_nau_vec, Lambda_au_vec = sim_Lambda(energy_vec_au, bath, adsorbate_model, adsorbate_position_vec_au, temp_au)
            push!(results, Dict(
                "temperature_au" => temp_au,
                "temperature" => ustrip(auconvert(u"K",temp_au)),
                "Lambda_nau_vec" => Lambda_nau_vec,
                "Lambda_au_vec" => Lambda_au_vec
            ))
        end
        return Dict("scalar_name" => "Position", "scalar_value" => scalar_value, "results" => results)

    # Check if position is a vector and temperature is a scalar
    elseif typeof(adsorbate_position_vec_au) <: AbstractVector && !(typeof(temperature_vec_au) <: AbstractVector)
        
        scalar_value = ustrip(auconvert(u"K",temperature_vec_au))
        results = []
        for pos_au in adsorbate_position_vec_au
            Lambda_nau_vec, Lambda_au_vec = sim_Lambda(energy_vec_au, bath, adsorbate_model, pos_au, temperature_vec_au)
            push!(results, Dict(
                "position_au" => pos_au,
                "position" => ustrip(auconvert(u"Å",pos_au)),
                "Lambda_nau_vec" => Lambda_nau_vec,
                "Lambda_au_vec" => Lambda_au_vec
            ))
        end
        return Dict("scalar_name" => "Temperature", "scalar_value" => scalar_value, "results" => results)

    else
        error("Either temperature_vec_au or adsorbate_position_vec_au must be a vector, but not both.")
    end
end




function plot_lambda_vs_energy()
    ## nice numerics for pretty plotting
    pretty(v) = sprint(io -> show(IOContext(io, :compact => true), "text/plain", v))
    ## Plotting set up
    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Energy / eV", ylabel= "Lambda / a.u.⁻²",limits=(nothing, nothing, nothing, nothing))

    # Get the structured results from the new batch function
    data = sim_Lambda_batch_single_vector(energies_au, adsorbate_m, position_au, temperatures_au)

    # Get the scalar value and its name for the title/group label
    scalar_name = data["scalar_name"]
    scalar_value = data["scalar_value"]
    results = data["results"]

    # Loop through the results to plot the lines
    for (i, result) in enumerate(results)
        Lambda_vec = result["Lambda_au_vec"]
        
        # Determine the label for the legend based on which parameter is the vector
        if haskey(result, "temperature")
            legend_label = "$(ceil(Int,result["temperature"])) K"
        else
            legend_label = "x = $(pretty(scalar_value)) Å"
        end

        lines!(ax, energies, Lambda_vec; color = colormap[i], linewidth = 3, label= legend_label)
    end
    
    # Add a title/label for the constant parameter
    # Using `latexstring` for nice formatting
    if scalar_name == "Position"
        title_text = "x = $(pretty(scalar_value)) Å"
    else
        title_text = "T = $(ceil(Int, scalar_value)) K"
    end
    
    Legend(fig[1,1], ax, title_text, titleposition = :top, tellwidth=false, tellheight=false, valign=:top, halign=:right, margin=(5, 5, 5, 5), orientation=:vertical)
    
    return fig 
end

save(plotsdir("frequency_friction", "Lambda_x=$(position)_nstates_$(nstates).pdf"), plot_lambda_vs_energy())









