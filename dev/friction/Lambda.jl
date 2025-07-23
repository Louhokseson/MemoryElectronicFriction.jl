using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end

using HokseonAssistant, HokseonPlots
using Unitful, UnitfulAtomic
using NQCModels
using NQCModels.DiabaticModels
using LinearAlgebra
using CairoMakie
using ColorSchemes
using Colors
using QuadGK
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;
HokseonAssistant.julia_session()

include("parameters.jl")


function singularities(bath, ω::Real)
    if ω == 0.0
        return sort(bath.bathstates)
    else
        return sort(vcat(bath.bathstates, bath.bathstates .- ω))
    end
end


function interval_limits(a::Real, b::Real, singularities, ϵ)
    """
    a: Lower limit of the interval
    b: Upper limit of the interval
    singularities: Vector of singularity points
    """
    a_new = a
    b_new = b
    if a in singularities
        a_new = a + ϵ  # Shift slightly to avoid singularity
    end
    if b in singularities
        b_new = b - ϵ  # Shift slightly to avoid singularity
    end
    return a_new, b_new
end

# Cauchy principal value integration
function principal_value_integral(f, ω::Real, bath; ε=1e-3)
    # Get all singularities in x₁
    sing_pts = singularities(bath,ω)

    # Integration intervals: split at each singularity
    bounds = [-Inf; sing_pts; Inf]

    total = 0.0
    total_error = 0.0

    for i in 1:length(bounds)-1
        a, b = interval_limits(bounds[i], bounds[i+1], sing_pts, ε)

        integral, error = quadgk(x1 -> f(x1, ω), a, b; rtol=1e-6)
        total += integral
        total_error += error
    end

    return total, total_error
end


function Lambda(energy::Real, bath, adsorbate_m::AndersonImpurityModel, position::Real ,temperature::Real, fermi_level::Real=0.0)
    """
    Lambda : Calculate the energy dependent friction 
             at a given energy, electronic temperature, discretised bath and adsorbate model.
    
    energy : Energy value
    bath : Bath object containing bath states and coupling constants
    adsorbate_m : AndersonImpurityModel object
    position : Position of the adsorbate from substrate
    temperature : Electronic temperature in atomic units
    fermi_level : Fermi level of the system in atomic units (default is 0.0 eV)
    
    Returns a scalar value representing the energy dependent friction at the given energy.
    """

    fermidirac = DistributionTools.FermiDirac(fermi_level, temperature)

    
    g(ω₁,ω₂) = AndersonImpurityFrictions.WeightedEHPDOS.Gamma(ω₁, ω₂, bath, adsorbate_m, position)
    f(ω₁, ω) = -1/ω * g(ω₁, ω + ω₁) * (HokseonReproduce.PDF.(ω + ω₁,fermidirac) - HokseonReproduce.PDF.(ω₁,fermidirac))

    Lambda_val, err = principal_value_integral(f, energy, bath; ε=1e-3)
    return Lambda_val
end

position = austrip(0.5*u"Å")


energies = (collect(1:500))
energies_au = austrip.(energies*u"eV")
#energy = austrip(10/1000 * u"eV")
temperature = austrip(300*u"K")
adsorbate_m = AndersonImpurityModels.BrandbygeAdsorbate()

Lambda_au = Lambda(energies_au[1], bath, adsorbate_m, position, temperature)

Lambda_fs⁻² = auconvert(u"fs^-2", Lambda_au)