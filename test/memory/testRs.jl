using HokseonReproduce
using Test


energy = 15
k = 3
adsorbate_m = AndersonImpurityModels.BrandbygeAdsorbate()
bathstates = collect(range(0, 6, length=150))  # Example bath states
coupling_k = 3.0


@testset "ImaginaryGreens (Raa)" begin
    position = 1.0  # Example position
    result_Raa = AndersonImpurityFrictions.ImaginaryGreens.Raa(energy, adsorbate_m, position)
    expected_Raa = HokseonReproduce.DOS(position, adsorbate_m)
    @test result_Raa ≈ HokseonReproduce.PDF(energy, expected_Raa) * 2pi
end

@testset "ImaginaryGreens (Rak)" begin

    ## Eq A48 comes from https://doi.org/10.1103/PhysRevB.52.6042
    position = 1.0  # Example position
    result_Rak = AndersonImpurityFrictions.ImaginaryGreens.Rak(energy, bathstates, k, adsorbate_m, position, coupling_k)
    
    Jₐ = HokseonReproduce.DOS(position, adsorbate_m)
    Δ = Jₐ.Γ  # Lorentzian width
    ϵ = Jₐ.ω0  # Lorentzian centre

    delta_dist_approx = DistributionTools.Gaussian(bathstates[k], 0.0001)

    Raa_value = AndersonImpurityFrictions.ImaginaryGreens.Raa(energy, adsorbate_m, position)

    A48b_bracket_left = Raa_value * (1/(energy - bathstates[k])) 
    A48b_bracket_right = Raa_value * (energy - ϵ) / (Δ) * pi * HokseonReproduce.PDF(energy, delta_dist_approx)

    expected_Rak = coupling_k * (A48b_bracket_left + A48b_bracket_right)

    @test result_Rak ≈ expected_Rak
end

@testset "ImaginaryGreens (Rkk′)" begin

    ## Eq 5.26 comes from Understanding of Green's function by Xuexun Lu

    position = 1.6  # Example position
    k′ = 5  # Another example index
    coupling_k′ = 5.0
    result_Rkk′ = AndersonImpurityFrictions.ImaginaryGreens.Rkk′(energy, bathstates, k, k′, adsorbate_m, position, coupling_k, coupling_k′)
    
    Jₐ = HokseonReproduce.DOS(position, adsorbate_m)
    Δ = Jₐ.Γ  # Lorentzian width
    ϵ = Jₐ.ω0  # Lorentzian centre

    delta_dist_approx_k = DistributionTools.Gaussian(bathstates[k], 0.0001)

    delta_dist_approx_k′ = DistributionTools.Gaussian(bathstates[k′], 0.0001)

    kronecker_delta = k == k′ ? 1.0 : 0.0

    first_term = kronecker_delta * 2pi * HokseonReproduce.PDF(energy, delta_dist_approx_k)

    """
    Im[G(ret)ak'] starts here
    """

    Raa_value = AndersonImpurityFrictions.ImaginaryGreens.Raa(energy, adsorbate_m, position)

    A48b_bracket_left = Raa_value * (1/(energy - bathstates[k′])) 
    A48b_bracket_right = Raa_value * (energy - ϵ) / (Δ) * pi * HokseonReproduce.PDF(energy, delta_dist_approx_k′)

    expected_Rak′ = coupling_k′ * (A48b_bracket_left + A48b_bracket_right)

    Im_Gret_ak′ = expected_Rak′ / 2 * -1

    """
    Im[G(ret)ak'] ends here
    """



    """
    Re[G(ret)ak'] starts here
    """
    ## Eq 5.25 comes from Understanding of Green's function by Xuexun Lu

    bracket_first = Raa_value * (energy - ϵ) / (Δ) * (1/(energy - bathstates[k′]))

    bracket_second = Raa_value * pi * HokseonReproduce.PDF(energy, delta_dist_approx_k′)

    Re_Gret_ak′ = - 0.5 * coupling_k′ * (bracket_first + bracket_second)

    """
    Re[G(ret)ak'] ends here
    """



    second_term = -2 * coupling_k * ( - Re_Gret_ak′ * π * HokseonReproduce.PDF(energy, delta_dist_approx_k) + Im_Gret_ak′ * (1/(energy - bathstates[k])) )


    expected_Rkk′ = first_term + second_term

    @info "result_Rkk′: $result_Rkk′"
    @info "expected_Rkk′: $expected_Rkk′"


    @test result_Rkk′ - expected_Rkk′ ≈ 0
end



@testset "Raa broadcastable" begin
    position = 1.0  # Example position

    energy_vec = range(0, 6, length=100)  # Example energy range

    result_Raa = AndersonImpurityFrictions.ImaginaryGreens.Raa.(energy_vec, adsorbate_m, position)
    adsorbate_lorentzian = HokseonReproduce.DOS(position, adsorbate_m)

    @test sum(result_Raa .≈ HokseonReproduce.PDF.(energy_vec, adsorbate_lorentzian) .* 2pi) == length(energy_vec)
end

@testset "Rak broadcastable" begin
    position = 1.0  # Example position

    energy_vec = range(0, 6, length=100)  # Example energy range
    k = 3  # Example index

    result_Rak = AndersonImpurityFrictions.ImaginaryGreens.Rak.(energy_vec, Ref(bathstates), Ref(k), Ref(adsorbate_m), Ref(position), Ref(coupling_k))
    
    Jₐ = HokseonReproduce.DOS(position, adsorbate_m)
    Δ = Jₐ.Γ  # Lorentzian width
    ϵ = Jₐ.ω0  # Lorentzian centre

    delta_dist_approx = DistributionTools.Gaussian(bathstates[k], 0.0001)

    Raa_value = AndersonImpurityFrictions.ImaginaryGreens.Raa.(energy_vec, Ref(adsorbate_m), Ref(position))

    A48b_bracket_left = Raa_value .* (1 ./(energy_vec .- bathstates[k])) 
    A48b_bracket_right = Raa_value .* (energy_vec .- ϵ) ./ (Δ) .* pi .* HokseonReproduce.PDF.(energy_vec, delta_dist_approx)

    expected_Rak = coupling_k .* (A48b_bracket_left .+ A48b_bracket_right)

    @test sum(result_Rak .≈ expected_Rak) == length(energy_vec)
end


@testset "Rkk′ broadcastable" begin
    position = 1.6  # Example position
    k = 3  # Example index
    k′ = 5  # Another example index
    coupling_k′ = 5.0
    energy_vec = range(0, 6, length=100)  # Example energy range

    result_Rkk′ = AndersonImpurityFrictions.ImaginaryGreens.Rkk′.(energy_vec, Ref(bathstates), Ref(k), Ref(k′), Ref(adsorbate_m), Ref(position), Ref(coupling_k), Ref(coupling_k′))
    
    Jₐ = HokseonReproduce.DOS(position, adsorbate_m)
    Δ = Jₐ.Γ  # Lorentzian width
    ϵ = Jₐ.ω0  # Lorentzian centre

    delta_dist_approx_k = DistributionTools.Gaussian(bathstates[k], 0.0001)

    delta_dist_approx_k′ = DistributionTools.Gaussian(bathstates[k′], 0.0001)

    kronecker_delta = k == k′ ? 1.0 : 0.0

    first_term = kronecker_delta .* 2pi .* HokseonReproduce.PDF.(energy_vec, delta_dist_approx_k)

    """
    Im[G(ret)ak'] starts here
    """

    Raa_value = AndersonImpurityFrictions.ImaginaryGreens.Raa.(energy_vec, Ref(adsorbate_m), Ref(position))

    A48b_bracket_left = Raa_value .* (1 ./(energy_vec .- bathstates[k′])) 
    A48b_bracket_right = Raa_value .* (energy_vec .- ϵ) ./ (Δ) .* pi .* HokseonReproduce.PDF.(energy_vec, delta_dist_approx_k′)

    expected_Rak′ = coupling_k′ .* (A48b_bracket_left .+ A48b_bracket_right)

    Im_Gret_ak′ = expected_Rak′ ./ 2 .* -1

    """
    Im[G(ret)ak'] ends here
    """

    """
    Re[G(ret)ak'] starts here
    """
    ## Eq 5.25 comes from Understanding of Green's function by Xuexun Lu

    bracket_first = Raa_value .* (energy_vec .- ϵ) ./ (Δ) .* (1 ./(energy_vec .- bathstates[k′]))
    bracket_second = Raa_value .* pi .* HokseonReproduce.PDF.(energy_vec, delta_dist_approx_k′)
    Re_Gret_ak′ = - 0.5 .* coupling_k′ .* (bracket_first .+ bracket_second)
    """
    Re[G(ret)ak'] ends here
    """
    second_term = -2 .* coupling_k .* ( - Re_Gret_ak′ .* π .* HokseonReproduce.PDF.(energy_vec, delta_dist_approx_k) .+ Im_Gret_ak′ .* (1 ./(energy_vec .- bathstates[k])) )
    expected_Rkk′ = first_term .+ second_term

    
    @test sum(result_Rkk′ .≈ expected_Rkk′) == length(energy_vec)
end


@info "Test of R terms in ImaginaryGreen.jl finished."