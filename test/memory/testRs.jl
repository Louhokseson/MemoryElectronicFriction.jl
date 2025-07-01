using HokseonReproduce
using Test


energy = 1
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