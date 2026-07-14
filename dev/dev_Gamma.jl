
"""
    dev_Gamma.jl
    
    This script is used for developing the Gamma function (Eq. A33) from Brandbyge's paper [1].
    
    [1] https://doi.org/10.1103/PhysRevB.52.6042

"""


using DrWatson
@quickactivate "MemoryElectronicFriction"

# making sure that MemoryElectronicFriction module is loaded once
if !isdefined(Main, :MemoryElectronicFriction)
    include(srcdir("MemoryElectronicFriction.jl"))
    using .MemoryElectronicFriction: TrapezoidalRule, ShenviGaussLegendre, BrandbygeModels, DistributionTools
end


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#                Bath's construction
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


include("dev_parameters.jl")
@unpack discretization, centre, width, nstates = bath_params

function build_WideBandBath!(discretization, centre, width, nstates)

    global bath, ρbath
    bandmin = centre - width/2
    bandmax = centre + width/2

    bath = eval(discretization)(nstates, bandmin, bandmax)

    # The density of states of the bath under WideBand approximation
    ρbath = nstates/width

end


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#            Bath's construction END
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX





#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#            Absorbate's construction
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

function build_BrandbygeAdsorbate!(distance)

    global Δ, eminiusH, lorentzian

    am = BrandbygeModels.BrandbygeAdsorbate()

    lorentzian = MemoryElectronicFriction.DOS(distance, am)

    Δ = lorentzian.Γ

    eminiusH = lorentzian.ω0

end



#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#            Absorbate's construction END
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX




#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#            coupling spectral funcions
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

"""
    Raa(ω):

    The projected density of states with respect to the absorbate.

    "aa" comes from the subscripts of Eq. A47 in Brandbyge's paper [1].

    [1] https://doi.org/10.1103/PhysRevB.52.6042

"""


function Raa(ω::Float64, distance::Float64)

    build_BrandbygeAdsorbate!(distance)

    return 2 * π * MemoryElectronicFriction.PDF(ω, lorentzian)
end 


"""
    Rak(ω,k):

    The spectral function of absorbate-bath state k coupling.

    "ak" comes from the subscripts of Eq. A48c in Brandbyge's paper [1].

    [1] https://doi.org/10.1103/PhysRevB.52.6042

    Tk: coupling strength

    first: the first term in the bracket in Eq. A48c

    second: the second term in the bracket in Eq. A48c
"""

function Rak(ω::Float64,k::Int64, distance::Float64)

    build_WideBandBath!(discretization, centre, width, nstates)

    states = bath.bathstates

    build_BrandbygeAdsorbate!(distance)

    if ω == states[k]
        error("Input frequency ω is equal to the k-th state energy.
              \n This generates an singularity in the expression.")
    end

    # build Dirac delta from a Gaussian
    DeltaGaussian = DistributionTools.Gaussian(states[k], 1e-3)

    first = Raa(ω, distance) * (1/(ω - states[k]))

    second = π * Raa(ω, distance) * MemoryElectronicFriction.PDF(ω, DeltaGaussian) * (ω-eminiusH)/Δ
    
    Tk = sqrt(Δ/(2*π*ρbath))

    return Tk * (first + second)
end

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#            coupling spectral funcions end
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX






