using DrWatson

@quickactivate "MemoryElectronicFriction"

using Unitful, UnitfulAtomic
using QuadGK
using CairoMakie
using HokseonPlots
using ColorSchemes
using Colors
using Unitful, UnitfulAtomic
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;

# making sure that MemoryElectronicFriction module is loaded only once
if !isdefined(Main, :MemoryElectronicFriction)
    include(srcdir("MemoryElectronicFriction.jl"))
    using .MemoryElectronicFriction
end



function ∂fermi(ϵ, μ, β)
    ∂f = -β * exp(β*(ϵ-μ)) / (1 + exp(β*(ϵ-μ)))^2
    return isnan(∂f) ? zero(ϵ) : ∂f
end


function widebandfriction(adsorbate_model::BrandbygeAdsorbate, r, μ, T)
    C = adsorbate_model.C
    α = adsorbate_model.α
    β = adsorbate_model.β
    Δ₀ = adsorbate_model.Δ₀
    ε∞ = adsorbate_model.ε∞


    h = ε∞ .- C .* exp.(-α .* r)
    Γ = Δ₀ .* exp.(-β .* r)

    dhdx = α .* C .* exp.(-α .* r)
    dΓdx = -β .* Δ₀ .* exp.(-β .* r)

    A(ϵ) = (1 / π) * (Γ ./ ((ϵ .- h).^2 .+ Γ.^2))
    
    kernel(ϵ) = -π * (dhdx + (ϵ .- h) .* dΓdx ./ Γ) .^ 2  .* A(ϵ).^2 * ∂fermi(ϵ, μ, 1/T)

    integral = quadgk(kernel, -Inf, Inf; rtol=1e-6)[1]

    return integral
end


all_params = Dict{String, Any}(
    "temperature"       => [300, 3500, 4000, 5000, 5500],
    "impuritymodel"     => :BrandbygeAdsorbate,
    "centre"            => [0],
    "position"          => [collect(0.0:0.01:4.0)],
)

params_list = dict_list(all_params)
# just make sure that params_list is a list with Dicts
if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end


function plot_widebandfriction()

    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Position / Å", ylabel= "Brandbyge Markovian friction (u⋅ps)⁻¹",limits=(1, 4, -1, 5))

    NO_mass_au = austrip(30.0057 * u"u") # mass of electron in atomic units

    for (i,param) in enumerate(params_list)

        @unpack temperature, impuritymodel, centre, position = param

        T = austrip(temperature*u"K")
        #β = 1/austrip(T*u"K")
        μ = austrip(centre * u"eV")
        r_au = austrip.(position .* u"Å")
        adsorbate_model = eval(impuritymodel)()


        markovian_friction_au = widebandfriction.(Ref(adsorbate_model), r_au, Ref(μ), Ref(T)) ./ NO_mass_au

        markovian_friction = markovian_friction_au ./ ustrip.(auconvert.(u"ps",1)) ./ ustrip.(auconvert.(u"u",1)) # converting from a.u. to (u ⋅ ps)⁻¹

        r = ustrip.(auconvert.(u"Å", r_au))

        lines!(ax, r, markovian_friction; color = colormap[i], linewidth = 2, label = "T = $(temperature) K")

    end
    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:right, margin=(5, 5, 5, 5), orientation=:vertical)

    return fig 
end

plot_widebandfriction()