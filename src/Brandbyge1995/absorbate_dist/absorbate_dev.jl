using DrWatson
@quickactivate "HokseonReproduce"

include(srcdir("distribution.jl"))
include(srcdir("show_model.jl"))

using CairoMakie
using CairoMakie
using JamesPlots
using ColorSchemes
using Colors
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = JamesPlots.NICECOLORS;



"""
build the abosorbate spaital distribution

"""


struct BrandbygeAborbateModel
    Δ₀::AbstractFloat
    β::AbstractFloat
    ε∞::AbstractFloat
    C::AbstractFloat
    α::AbstractFloat
end
function BrandbygeAborbateModel(;
        Δ₀= 0.2,
        β= 1.0,
        ε∞= 5.0,
        C= 3.0,
        α = 0.5,
    )
    return BrandbygeAborbateModel(Δ₀, β, ε∞, C, α)
end


function DOS(r::Real ,Δϵ::Real, m::BrandbygeAborbateModel)
    """
    DOS of the impurity follows the equation (22 a,b)

    we assume the absorbate is a Cauchy/Lorentz distribution

    r: distance from the impurity to the reservoir

    Δϵ: energy range for plotting the absorbate DOS
    
    m: BrandbygeAborbateModel
    """
    Δ = m.Δ₀ * exp(-m.β * r)
    ϵₐ = m.ε∞ - m.C * exp(-m.α * r)

    lorentzian(ω) = Lorentzian(ω, ϵₐ, Δ)

    dos_r = collect(range(ϵₐ - Δϵ, ϵₐ + Δϵ, length = 100))

    dos = lorentzian.(dos_r)
    return dos_r, dos
end



# Example usage


function plot_brandbyge_dos(r = 1.0, Δϵ = 1.0)
    m = BrandbygeAborbateModel()
    dos_r, dos = DOS(r, Δϵ, m)
    ## Plotting set up
    fig = Figure(size=(JamesPlots.RESOLUTION[1]*2, 3*JamesPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Energy / eV", ylabel= "DOS",limits=(nothing, nothing, nothing, nothing))


    lines!(ax, dos_r, dos; color = colormap[1], linewidth = 2, label = "Brandbyge \n impurity at $r")
    band!(ax, dos_r, zeros(length(dos_r)), dos; color=(colormap[1],0.3))
    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)
    return fig
end

plot_brandbyge_dos(5, 1.0)