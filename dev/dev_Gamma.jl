
"""
    dev_Gamma.jl
    
    This script is used for developing the Gamma function (Eq. A33) from Brandbyge's paper [1].
    
    [1] https://doi.org/10.1103/PhysRevB.52.6042

"""


using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce: TrapezoidalRule, ShenviGaussLegendre, BrandbygeModels, DistributionTools
end


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#                Bath's construction
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


include("dev_parameters.jl")
@unpack discretization, centre, width, nstates = bath_params

bandmin = centre - width/2
bandmax = centre + width/2

bath = eval(discretization)(nstates, bandmin, bandmax)

states = bath.bathstates

# The density of states of the bath under WideBand approximation
ρbath = nstates/width


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#            Bath's construction END
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX





#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#            Absorbate's construction
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


distance = 1.0

am = BrandbygeModels.BrandbygeAbsorbate()

lorentzian = HokseonReproduce.DOS(distance, am)

Δ = lorentzian.Γ

eminiusH = lorentzian.ω0


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


Raa(ω) =  2 * π * HokseonReproduce.PDF(ω, lorentzian)


"""
    Rak(ω,k):

    The spectral function of absorbate-bath state k coupling.

    "ak" comes from the subscripts of Eq. A48c in Brandbyge's paper [1].

    [1] https://doi.org/10.1103/PhysRevB.52.6042

    Tk: coupling strength

    first: the first term in the bracket in Eq. A48c

    second: the second term in the bracket in Eq. A48c
"""

function Rak(ω::Float64,k::Int64)
    if ω == states[k]
        error("Input frequency ω is equal to the k-th state energy.
              \n This generates an singularity in the expression.")
    end

    # build Dirac delta from a Gaussian
    DeltaGaussian = DistributionTools.Gaussian(states[k], 1e-3)

    first = Raa(ω) * (1/(ω - states[k]))

    second = π * Raa(ω) * HokseonReproduce.PDF(ω, DeltaGaussian) * (ω-eminiusH)/Δ
    
    Tk = sqrt(Δ/(2*π*ρbath))

    return Tk * (first + second)
end

Rak(10.0, 50)




