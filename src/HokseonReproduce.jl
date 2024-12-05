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


# Define the top-level DOS function
function DOS(r::Real, Δϵ::Real, m::Model)
    error("DOS function not implemented for this model type")
end

function PDF(ω::Real, d::Distribution)
    error("PDF function not implemented for this distribution type")
end


# Importing the submodules [loads the independent modules first]
# BrandbygeModels depends on Distributions
include("DistributionsTools/Distributions.jl")
@reexport using .Distributions 

include("Brandbyge1995/BrandbygeModels.jl")
@reexport using .BrandbygeModels


# @reexport makes sure that you could call Distributions.function() directly
# without having to call HokseonReproduce.Distributions.function()
end # module