module Tools
    
# Importing the parent module
using ..MemoryElectronicFriction: MemoryElectronicFriction 

# Importing the external packages
using DrWatson

abstract type Tool end

include("MathsTools/distribution.jl")
export FermiDirac

end