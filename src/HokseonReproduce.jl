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


# Define the top-level DOS function
function DOS(r::Real, Δϵ::Real, m::Model)
    error("DOS function not implemented for this model type")
end

# Importing the submodules
include("Brandbyge1995/BrandbygeModels.jl")
@reexport using .BrandbygeModels


end # module