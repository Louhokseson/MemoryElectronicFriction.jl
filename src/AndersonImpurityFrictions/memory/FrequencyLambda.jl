module FrequencyLambda
import ..HokseonReproduce, ..DistributionTools, ..AndersonImpurityModel, ..WeightedEHPDOS
using QuadGK

function singularities(bath, ω::Real)
    if ω == 0.0
        return sort(bath.bathstates)
    else
        ## {bath.bathstates, bath.bathstates - ω} union set
        sing = Set(Iterators.flatten((bath.bathstates, bath.bathstates .- ω)))
        return sort!(collect(sing))
    end
end

function interval_limits(a::Real, b::Real, singularities, ϵ_shift)
    """
    a: Lower limit of the interval
    b: Upper limit of the interval
    ## a < b
    singularities: Vector of singularity points
    """
    a_new = a
    b_new = b
    if a in singularities
        a_new = a + ϵ_shift  # Shift slightly to avoid singularity
    end
    if b in singularities
        b_new = b - ϵ_shift  # Shift slightly to avoid singularity
    end
    return a_new, b_new
end

# Cauchy principal value integration
function principal_value_integral(f, ω::Real, bath; ε=1e-4)
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

    Reference: A37 in paper https://doi.org/10.1103/PhysRevB.52.6042
    
    energy : Energy value
    bath : Bath object containing bath states and coupling constants
    adsorbate_m : AndersonImpurityModel object
    position : Position of the adsorbate from substrate
    temperature : Electronic temperature in atomic units
    fermi_level : Fermi level of the system in atomic units (default is 0.0 eV)
    
    Returns a scalar value representing the energy dependent friction at the given energy.
    """

    fermidirac = DistributionTools.FermiDirac(fermi_level, temperature)

    
    Γ(ω₁,ω₂) = WeightedEHPDOS.Gamma(ω₁, ω₂, bath, adsorbate_m, position)
    f(ω₁, ω) = Γ(ω₁, ω + ω₁) * (HokseonReproduce.PDF.(ω + ω₁,fermidirac) - HokseonReproduce.PDF.(ω₁,fermidirac))

    integral_val, err = principal_value_integral(f, energy, bath; ε=1e-3)
    return - 1/energy * integral_val / (2π)
end


# New vector overload
function Lambda(energy_vec::AbstractVector, bath, adsorbate_m::AndersonImpurityModel,
                position::Real, temperature::Real, fermi_level::Real=0.0)

    if Threads.nthreads() == 1
        # Single-thread → broadcast
        return Lambda.(energy_vec, Ref(bath), Ref(adsorbate_m),
                       Ref(position), Ref(temperature), Ref(fermi_level))
    else
        # Multi-threaded loop
        Lambda_au_vec = Vector{Float64}(undef, length(energy_vec))
        Threads.@threads for i in eachindex(energy_vec)
            Lambda_au_vec[i] = Lambda(energy_vec[i], bath, adsorbate_m,
                                      position, temperature, fermi_level)
        end
        return Lambda_au_vec
    end
end


end