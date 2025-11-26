module FrequencyLambda
import ..HokseonReproduce, ..DistributionTools, ..AndersonImpurityModel, ..WeightedEHPDOS
using ..DistributionTools: FermiDirac
using QuadGK

function bath_singularities(bath, ω::Real)

    """
    singularities 奇异值点来自于 bath 自身的离散能级 + bath 能级减去给定 ω 的值的并集
    """
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
function principal_value_integral(f, ω::Real, sing_pts; ε=1e-9)
    # Get all singularities in x₁
    #sing_pts = singularities(bath,ω)

    # Integration intervals: split at each singularity
    bounds = [-Inf; sing_pts; Inf]

    total = 0.0
    total_error = 0.0

    for i in 1:length(bounds)-1
        a, b = interval_limits(bounds[i], bounds[i+1], sing_pts, ε)

        integral, error = quadgk(x1 -> f(x1, ω), a, b; rtol=1e-12)
        total += integral
        total_error += error
    end

    return total, total_error
end


function ω₁_effective_range(fermidirac::FermiDirac, ω::Real; C::Real = 6)
    """
         [nf(ω+ω₁) - nf(ω₁)] 结果非零的有效范围 (给定 ω 的情况下， ω₁ 的范围)
    
    ω₁ : 能量变量 ω₁
    ω  : 给定的能量值 ω
    C  : 自己可以调的常数，决定非0范围的宽度 -- 类似于置信区间的常数选择

    返回 : ω₁ 的有效范围 (ω₁_min, ω₁_max)
    """
    if ω >= 0
        # Positive ω case
        ω₁_min = fermidirac.ϵf - ω - C * fermidirac.T
        ω₁_max = fermidirac.ϵf + C * fermidirac.T
    else
        # Negative ω case
        ω₁_min = fermidirac.ϵf - C * fermidirac.T
        ω₁_max = fermidirac.ϵf - ω + C * fermidirac.T
    end

    return (ω₁_min, ω₁_max)
end

function piecewise_midpoint_cauchy_interval(sing_pts::AbstractArray)


    """
    midpoints: 提供的奇异点之间的中点列表

    """

    midpoints = vcat([(sing_pts[1] - (sing_pts[2] - sing_pts[1]) + sing_pts[1])/2], 
                             ([(sing_pts[i] + sing_pts[i+1])/2] for i in 1:(length(sing_pts)-1))..., 
                             [(sing_pts[end] + (sing_pts[end] + (sing_pts[end] - sing_pts[end-1])))/2])


    return midpoints

end

function effective_bounds(bath, fermidirac::FermiDirac, ω::Real)

    """

    bath       : 离散的 bath 对象
    fermidirac : FermiDirac 分布对象
    ω          : 能量值
    midpoints  : 奇异值之间的中点

    返回        : ω₁的有效范围里的奇异点和中点

    """

    bath_sing_pts = bath_singularities(bath,ω)

    ω₁_range_min, ω₁_range_max = ω₁_effective_range(fermidirac, ω)

    midpoints = piecewise_midpoint_cauchy_interval(bath_sing_pts)

    bounds = sort([ω₁_range_min; ω₁_range_max; bath_sing_pts]) |> x -> x[min(ω₁_range_min,ω₁_range_max) .≤ x .≤ max(ω₁_range_min,ω₁_range_max)]

    #sing_pts_eff = bath_sing_pts[min(ω₁_range_min,ω₁_range_max) .≤ bath_sing_pts .≤ max(ω₁_range_min,ω₁_range_max)]

    return bounds
end




function cauchy_integral(f, ω::Real, bounds::AbstractArray; ε=1e-7)

    """
    分段积分   : 根据奇异点之间的中点将积分区间划分为多个子区间，在每个子区间上进行积分

    f。      : 被积函数
    ω        : 能量值
    bounds   : 分段积分的区间边界
    ε        : 用于避开奇异点的微小偏移量
    """

    total = 0.0

    for i in 1:length(bounds)-1
        a, b = bounds[i], bounds[i+1]

        integral = 0.0
        try
            integral = quadgk(x1 -> f(x1, ω), a + ε, b - ε; rtol=1e-12)[1]
        catch e
            @error "Piecewise integration failed in interval [$a, $b]: \n $e" 
        end
        total += integral
    end

    return total

end


function Lambda(energy::Real, bath, adsorbate_m::AndersonImpurityModel, position::Real ,temperature::Real, fermi_level::Real=0.0)
    """
    Lambda : Calculate the energy dependent friction 
             at a given energy, electronic temperature, discretised bath and adsorbate model.

    Reference: A37 in paper https://doi.org/10.1103/PhysRevB.52.6042
    
    energy : Energy value ω
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

    bounds = effective_bounds(bath, fermidirac, energy)
    
    integral_val = cauchy_integral(f, energy, bounds)
    return - 1/energy * integral_val / (2π)
end


function Lambda(energy_vec::AbstractVector, bath, adsorbate_m::AndersonImpurityModel,
                position::Real, temperature::Real, fermi_level::Real=0.0)

    Lambda_au_vec = similar(energy_vec, Float64)

    @info "Using $(Threads.nthreads()) threads for frequency dependent friction Λ(ω) calculation"

    Threads.@threads for i in eachindex(energy_vec)
        @inbounds Lambda_au_vec[i] = Lambda(energy_vec[i], bath, adsorbate_m, position, temperature, fermi_level)
    end

    return Lambda_au_vec
end



end