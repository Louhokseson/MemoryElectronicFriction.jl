module HokseonReproduce

using Reexport: @reexport


"""
Top-level type for models.

# Implementation
When adding new models, this should not be directly subtyped. Instead, depending on
the intended functionality of the model, one of the child abstract types should be
subtyped.
If an appropriate type is not already available, a new abstract subtype should be created. 
"""
abstract type Model end
Base.broadcastable(model::Model) = Ref(model)

abstract type Friction end
Base.broadcastable(friction::Friction) = Ref(friction)

abstract type Distribution end
Base.broadcastable(dist::Distribution) = Ref(dist)

abstract type Bath end
Base.broadcastable(bath::Bath) = Ref(bath)

# Define the top-level DOS function
function DOS(r::Real, Δϵ::Real, m::Model)
    error("DOS function not implemented for this model type")
end

function ϵₐ(r::Real, ω::Real, m::Model)
    error("ϵₐ function not implemented for this model type")
end

function Δ(r::Real, ω::Real, m::Model)
    error("Δ function not implemented for this model type")
end

function adsorbate_h(r::Real,m::Model)
    error("adsorbate_h function not implemented for this model type")
end

function adsorbate_diabatic(r::Real,m::Model)
    error("adsorbate_diabatic function not implemented for this model type")
end

function dh_dr(r::Real,m::Model)
    error("dh_dr function not implemented for this model type")
end

function dh_dx(r::Real,m::Model)
    error("dh_dx function not implemented for this model type")
end

function coupling_A(r::Real,m::Model)
    error("coupling_A function not implemented for this model type")
end

function dA_dr(r::Real,m::Model)
    error("dA_dr function not implemented for this model type")
end

function PDF(ω::Real, d::Distribution)
    error("PDF function not implemented for this distribution type")
end

function dΔ_dr(r::Real, d::Distribution)
    error("Δ_dr function not implemented for this distribution type")
end

function dΔ_dx(r::Real, d::Distribution)
    error("Δ_dx function not implemented for this distribution type")
end

function dϵₐ_dr(r::Real, d::Distribution)
    error("ϵₐ_dr function not implemented for this distribution type")
end

function dΔ_dr(r::Real, energy::Real, d::Distribution)
    error("Δ_dr function not implemented for this distribution type")
end

function dϵₐ_dr(r::Real, energy::Real, d::Distribution)
    error("ϵₐ_dr function not implemented for this distribution type")
end

function Ak(bath::Bath, adsorbate_m::Model, position::Real)
    error("Ak function not implemented for this bath and adsorbate model")
end

function Vak(bath::Bath, adsorbate_m::Model, position::Real)
    error("Vak function not implemented for this bath and adsorbate model")
end

function V′ak(bath::Bath, adsorbate_m::Model, position::Real)
    error("V′ak function not implemented for this bath and adsorbate model")
end


# Importing the submodules [loads the independent modules first]
# BrandbygeModels depends on Distributions
include("DistributionTools/DistributionTools.jl")
@reexport using .DistributionTools 

include("Baths/Baths.jl")
@reexport using .Baths

include("AndersonImpurityModels/AndersonImpurityModels.jl")
@reexport using .AndersonImpurityModels


include("AndersonImpurityFrictions/AndersonImpurityFrictions.jl")
@reexport using .AndersonImpurityFrictions


include("IO/IO.jl")
@reexport using .IO

# @reexport makes sure that you could call Distributions.function() directly
# without having to call HokseonReproduce.Distributions.function()
end # module