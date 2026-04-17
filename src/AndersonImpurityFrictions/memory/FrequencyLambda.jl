module FrequencyLambda
import ..HokseonReproduce, ..DistributionTools, ..WeightedEHPDOS
using ..DistributionTools: FermiDirac
using ..AndersonImpurityModels: WideBandLimitModel, WideBandLimitModel1DOF, WideBandLimitModelNDOF, FrequencyDependentModel, FrequencyDependentModel1DOF, BrandbygeAdsorbate, AndersonImpurityModel, AndersonImpurityModel1DOF, AndersonImpurityModelNDOF
using StaticArrays: SVector
using LinearAlgebra: Symmetric
using QuadGK
using Distributed # Needed for the pmap/workers logic
using FLoops

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

function effective_sing_pts(bath, fermidirac::FermiDirac, ω::Real)

    """

    bath       : 离散的 bath 对象
    fermidirac : FermiDirac 分布对象
    ω          : 能量值

    返回        : ω₁的有效范围里的奇异点

    """

    bath_sing_pts = bath_singularities(bath,ω)

    ω₁_range_min, ω₁_range_max = ω₁_effective_range(fermidirac, ω)

    bath_sing_pts_effective = bath_sing_pts[min(ω₁_range_min,ω₁_range_max) .≤ bath_sing_pts .≤ max(ω₁_range_min,ω₁_range_max)]

    return bath_sing_pts_effective
end



function find_effective_sing_pts_with_poles(sing_pts, f; peturbation::Real = 5e-7)
    """
    sing_pts : 原始奇异点列表
    f        : 被积函数
    peturbation : 用于在奇异点的peturbation
    
    返回      : 经过peturbation后非0的奇异点的列表
    """
    
    peturbated_sing_pts = sing_pts .- peturbation
    n = length(sing_pts)
    
    # 预分配数组
    mask = Vector{Bool}(undef, n)
    
    # 使用 @floop 填充数组
    @floop ThreadedEx() for i in 1:n
        mask[i] = f(peturbated_sing_pts[i]) > 5e-4
    end
    #@info sum(mask) == n
    
    return sing_pts[mask]
end



function cauchy_integral(f, ω, bounds)

    """
    分段积分   : 根据奇异点之间的中点将积分区间划分为多个子区间，在每个子区间上进行积分

    f。      : 被积函数
    ω        : 能量值
    bounds   : 分段积分的区间边界
    ε        : 用于避开奇异点的微小偏移量

    适用于两种情况

    一 : Lambda 已经被多线程调用 (一列energy_vec), 此时 cauchy_integral 使用单线程的方式进行积分计算
                线程1   单独地处理所有分段积分 @floop ThreadedEx() 
                其他线程 单独地处理所有分段积分 if Threads.threadid() > 1

    二 : Lambda 只在主线程中被调用 (单个energy), 此时 cauchy_integral 使用多线程的方式进行积分计算
                所有线程 多线程分配分段积分 @floop ThreadedEx() 
    """

    #@info "Thread ID: $(Threads.threadid()) starting cauchy integral computation."

    # If called inside a threaded region, use single-threaded fallback:
    if Threads.threadid() > 1
        total = 0.0
        for i in 1:length(bounds)-1
            a, b = bounds[i], bounds[i+1]

            Δ = b - a
            ε = max(Δ*1e-8, 1e-9)  # safer if interval is tiny

            total += quadgk(x -> f(x, ω), a + ε, b - ε)[1]
        end
        return total
    end

    # If called from main thread (scalar Lambda), use multithreading:
    @floop ThreadedEx() for i in 1:length(bounds)-1
        a, b = bounds[i], bounds[i+1]

        Δ = b - a
        ε = max(Δ*1e-8, 1e-9)  # safer if interval is tiny

        integral = quadgk(x -> f(x, ω), a + ε, b - ε)[1]
        @reduce total += integral
    end

    return total
end

function singularities_integral(f, ω, sing_pts; ϵ::Real = 1e-15, δ::Real=1e-6, adaptive::Bool = true)

    """
    分段积分   : 在奇异点周围建立分段积分区间，去掉奇异点本身

    f。        : 被积函数
    ω          : 能量值
    sing_pts   : 奇异点列表
    ϵ          : 用于避开奇异点的微小偏移量
    δ          : 用于定义奇异点周围积分区间的微小范围

    适用于两种情况

    一 : Lambda 已经被多线程调用 (一列energy_vec), 此时 singularities_integral 使用单线程的方式进行积分计算
                线程1   单独地处理所有分段积分 @floop ThreadedEx() 
                其他线程 单独地处理所有分段积分 if Threads.threadid() > 1

                **小结 : 每个线程被分配到一个energy, 然后各自处理所有奇异点的分段积分

    二 : Lambda 只在主线程中被调用 (单个energy), 此时 singularities_integral 使用多线程的方式进行积分计算
                所有线程 多线程分配分段积分 @floop ThreadedEx() 
                
                **小结 : 那单独的energy的分段积分被多个线程分配处理 (这一般分配了不同的energy到不同的worker进程中)
    """
    ## make sure δ is not too large compared to the spacing of singularities
    #δ = min(δ, minimum(diff(sort(sing_pts))) / 2)

    # 定义安全的积分函数
    function safe_quadgk(f, a, b)
        try
            return quadgk(f, a, b)
        catch e
            if adaptive && ϵ < 1e-10  
                @warn "Integration failed near [$a, $b] with ϵ=$ϵ. Trying with larger ϵ."
                # 递归调用但使用更大的ϵ
                new_ϵ = min(ϵ * 100, δ/10)
                return singularities_integral(f, ω, sing_pts; 
                                             ϵ=new_ϵ, δ=δ, 
                                             adaptive=false)  # 防止无限递归
            else
                rethrow(e)
            end
        end
    end

    @info "Thread ID: $(Threads.threadid()) starting cauchy integral computation."

    # If called inside a threaded region, use single-threaded fallback:
    if Threads.threadid() > 1
        total = 0.0
        total_error = 0.0
        for sing in sing_pts
            a, b = sing - δ, sing - ϵ
            A, B = sing + ϵ, sing + δ

            val1, err1 = safe_quadgk(x -> f(x, ω), a, b)
            val2, err2 = safe_quadgk(x -> f(x, ω), A, B)

            total += val1 + val2
            total_error += err1 + err2
        end
        @info "Estimated total integration error: $total_error"
        return total
    end

    # If called from main thread (scalar Lambda), use multithreading:
    @floop ThreadedEx() for sing in sing_pts
        a, b = sing - δ, sing - ϵ
        A, B = sing + ϵ, sing + δ

        val1, err1 = safe_quadgk(x -> f(x, ω), a, b)
        val2, err2 = safe_quadgk(x -> f(x, ω), A, B)
        integral = val1 + val2
        error = err1 + err2

        @reduce total += integral
        @reduce total_error += error
    end

    @info "Estimated total integration error: $total_error"

    return total
end

# --------------------------------------------------------------------------
# 2. SCALAR KERNEL (Unchanged - The core unit of work)
# --------------------------------------------------------------------------

function Lambda(energy::Real, bath, adsorbate_m::AndersonImpurityModel, position::Real ,temperature::Real, fermi_level::Real=0.0; ϵ_shift::Real=1e-13, find_sing_pts_poles::Bool = true)
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

    ϵ_shift : Small shift to avoid singularities in integration (default is 1e-13)
    
    Returns a scalar value representing the energy dependent friction at the given energy.
    """

    fermidirac = DistributionTools.FermiDirac(fermi_level, temperature)

    
    Γ(ω₁,ω₂) = WeightedEHPDOS.Gamma(ω₁, ω₂, bath, adsorbate_m, position)
    f(ω₁, ω) = Γ(ω₁, ω + ω₁) * (HokseonReproduce.PDF.(ω + ω₁,fermidirac) - HokseonReproduce.PDF.(ω₁,fermidirac))

    #bounds = effective_bounds(bath, fermidirac, energy)
    sing_pts = effective_sing_pts(bath, fermidirac, energy)

    if find_sing_pts_poles
        g(ω₁) = Γ(ω₁, energy + ω₁)
        sing_pts = find_effective_sing_pts_with_poles(sing_pts, g)
    end
    #integral_val = cauchy_integral(f, energy, bounds)
    integral_val = singularities_integral(f, energy, sing_pts; ϵ = ϵ_shift)
    return - 1/energy * integral_val / (2π)
end


# --------------------------------------------------------------------------
# 3. VECTOR OVERLOAD (The new automatic hybrid dispatcher)
# --------------------------------------------------------------------------

function Lambda(energy_vec::AbstractVector, bath, adsorbate_m::AndersonImpurityModel,
                position::Real, temperature::Real, fermi_level::Real=0.0; ϵ_shift::Real=1e-13, find_sing_pts_poles::Bool = true)
    
    # 1. Check available worker processes
    avail_workers = workers()
    # A robust check: if there is more than just the main process (PID 1)
    has_workers = length(avail_workers) > 1 || (length(avail_workers) == 1 && avail_workers[1] != 1)
    n_workers = length(avail_workers)

    if has_workers
        # ----------------------------------------------------
        # 🚀 STRATEGY 1: MULTIPROCESSING (Hybrid Outer/Inner)
        # ----------------------------------------------------

        # Get the number of threads available to each worker (which is the same as the Master's setting)
        threads_per_worker = Threads.nthreads()
        
        @info "Hybrid Parallelism: Distributing $(length(energy_vec)) energies across $n_workers worker processes, with $threads_per_worker threads per worker." 

        # Define the work function. This closure captures all arguments (bath, model, etc.)
        # and transfers them once per worker via pmap's internal mechanism.
        worker_func(e) = Lambda(e, bath, adsorbate_m, position, temperature, fermi_level; ϵ_shift, find_sing_pts_poles)

        # pmap distributes the work element-by-element. Each element executes Lambda(e),
        # where cauchy_integral uses the worker's internal threads.

        results_iterator = pmap(worker_func, energy_vec)
        
        return collect(results_iterator) # Collect the results into a single Vector

    else
        # -----------------------------------------------------------------
        # ⚙️ STRATEGY 2: FALLBACK TO LOCAL MULTITHREADING (Original optimized code)
        # -----------------------------------------------------------------
        n_energy = length(energy_vec)
        Lambda_au_vec = similar(energy_vec, Float64)

        @warn "No multiprocessing detected. Falling back to local multithreading ($(Threads.nthreads()) threads)."

        # --- Re-implementing the original efficient threaded loop ---
        # 1. Prebind Globals for the threaded loop
        local_bath   = bath
        local_model  = adsorbate_m
        local_pos    = position
        local_temp   = temperature
        local_ef     = fermi_level
        local_FD     = DistributionTools.FermiDirac(local_ef, local_temp)
        local_PDF    = HokseonReproduce.PDF
        local_Gamma  = (ω1, ω2) -> WeightedEHPDOS.Gamma(ω1, ω2, local_bath, local_model, local_pos)
        
        # 2. Precompile/Warm-up
        @info "Precompiling a single energy point before multithreading..."
        Lambda_au_vec[1] = Lambda(energy_vec[1], bath, adsorbate_m, position, temperature, fermi_level; find_sing_pts_poles)
        @info "Precompilation done."

        # 3. Threaded Loop (Outer parallelism). Inner integral will run serially.
        @info "Starting multithreaded Λ(ω) calculation..."
        Threads.@threads for i in 2:n_energy
            e = energy_vec[i]
            
            f = (ω1, ω) -> begin
                Γval = local_Gamma(ω1, ω1 + ω)
                dfdE = local_PDF(ω1 + ω, local_FD) - local_PDF(ω1, local_FD)
                Γval * dfdE
            end
            
            #bounds = effective_bounds(local_bath, local_FD, e)
            sing_pts = effective_sing_pts(local_bath, local_FD, e)
            if find_sing_pts_poles
                g(ω₁) = local_Gamma(ω₁, e + ω₁)
                sing_pts = find_effective_sing_pts_with_poles(sing_pts, g)
            end 
            integral = singularities_integral(f, e, sing_pts; ϵ = ϵ_shift)
            
            @inbounds Lambda_au_vec[i] = -(integral)/(e * 2π)
        end
        @info "Λ(ω) calculation completed."

        return Lambda_au_vec
    end
end


function Lambda(ω::Real, adsorbate_m::WideBandLimitModel1DOF, position::Real ,temperature::Real, fermi_level::Real=0.0)

    """

        Frequency-dependent friction calculation based on Anderson impurity model parameters
    
        输入:
        ω : 能量值 度量能量尺度
        position : adsorbate 与 substrate 的距离
        temperature : 电子温度
        fermi_level : 费米能级

    """

    r = position
    fermidirac = DistributionTools.FermiDirac(fermi_level, temperature)

    h = HokseonReproduce.adsorbate_h(r, adsorbate_m)
    Δ = HokseonReproduce.Δ(r, adsorbate_m)

    dϵₐdx = HokseonReproduce.dϵₐ_dr(r, adsorbate_m)
    dΔdx = HokseonReproduce.dΔ_dr(r, adsorbate_m)

    A(ϵ) =  2 .* Δ ./ ((ϵ .- h).^2 .+ Δ.^2)

    kernel(ω₁) = A(ω₁) .* A(ω + ω₁) .* (dϵₐdx + (ω₁ .- h) .* dΔdx ./ Δ) .* (dϵₐdx + (ω₁ .+ ω .- h) .* dΔdx ./ Δ) .* (HokseonReproduce.PDF.(ω + ω₁,fermidirac) - HokseonReproduce.PDF.(ω₁,fermidirac))

    integral = quadgk(kernel, -Inf, Inf; rtol=1e-6)[1]

    return - integral ./ (ω * 4π)

end

# Kernel for the (k,l) + (l,k) symmetrised integrand.
# Scalars dh_k, dh_l, dΔ_k, dΔ_l are the k-th and l-th components of the
# gradient vectors — passed as scalars to avoid per-quadgk-call allocation.
function Lambdaₖₗₗₖ(ω₁::Real, ω::Real, h::Real, Δ::Real,
                     dh_k::Real, dh_l::Real, dΔ_k::Real, dΔ_l::Real,
                     fermidirac::FermiDirac)
    Jₐ(ϵ) = 2Δ / ((ϵ - h)^2 + Δ^2)

    Fₖ(ω̃) = dh_k + (ω̃ - h) * dΔ_k / Δ
    Fₗ(ω̃) = dh_l + (ω̃ - h) * dΔ_l / Δ

    Δf = HokseonReproduce.PDF(ω + ω₁, fermidirac) - HokseonReproduce.PDF(ω₁, fermidirac)

    return Jₐ(ω₁) * Jₐ(ω + ω₁) * (Fₖ(ω₁) * Fₗ(ω₁ + ω) + Fₗ(ω₁) * Fₖ(ω₁ + ω)) * Δf / 4π
end

function Lambda(ω::Real, adsorbate_m::WideBandLimitModelNDOF, configuration::SVector, temperature::Real, fermi_level::Real=0.0)

    """
        
        Frequency-dependent friction calculation based on Anderson impurity model parameters
    
        输入:
        ω : 能量值 度量能量尺度
        configuration : adsorbate 的物理位置
        temperature : 电子温度
        fermi_level : 费米能级
    """

    ndof = adsorbate_m.ndof
    Lambda_mat = zeros(ndof, ndof)

    fermidirac = DistributionTools.FermiDirac(fermi_level, temperature)
    h = HokseonReproduce.adsorbate_h(configuration, adsorbate_m)
    Δ = HokseonReproduce.Δ(configuration, adsorbate_m)

    dhdx_vec = HokseonReproduce.dh_dx(configuration, adsorbate_m)
    dΔdx_vec = HokseonReproduce.dΔ_dx(configuration, adsorbate_m)

    for k in 1:ndof
        for l in k:ndof
            Lambda_mat[k, l] = quadgk(
                ω₁ -> Lambdaₖₗₗₖ(ω₁, ω, h, Δ, dhdx_vec[k], dhdx_vec[l], dΔdx_vec[k], dΔdx_vec[l], fermidirac),
                -Inf, Inf; rtol=1e-6)[1]
        end
    end

    return Symmetric(Lambda_mat) ./ (ω * -2)
end


function Lambda(ω::Real, adsorbate_m::FrequencyDependentModel1DOF, position::Real ,temperature::Real, fermi_level::Real=0.0)

    """

        Frequency-dependent friction calculation based on Anderson impurity model parameters
    
        输入:
        ω : 能量值 度量能量尺度
        position : adsorbate 与 substrate 的距离
        temperature : 电子温度
        fermi_level : 费米能级

    """

    r = position
    fermidirac = DistributionTools.FermiDirac(fermi_level, temperature)

    ϵₐ(ω₁) = HokseonReproduce.ϵₐ(r, ω₁, adsorbate_m)
    Δ(ω₁) = HokseonReproduce.Δ(r, ω₁, adsorbate_m)

    A = HokseonReproduce.coupling_A(r, adsorbate_m)
    dAdr = HokseonReproduce.dA_dr(r, adsorbate_m)
    dhdx = HokseonReproduce.dh_dr(r, adsorbate_m)
    h = HokseonReproduce.adsorbate_h(r, adsorbate_m)

    Jₐ(ϵ) =  2 .* Δ(ϵ) ./ ((ϵ .- ϵₐ(ϵ)).^2 .+ Δ(ϵ).^2)

    kernel(ω₁) = Jₐ(ω₁) .* Jₐ(ω + ω₁) .* (dhdx + (ω₁ .- h) .* 2 ./ A .* dAdr) .* (dhdx + (ω₁ + ω .- h) .* 2 ./ A .* dAdr) .* (HokseonReproduce.PDF.(ω + ω₁,fermidirac) - HokseonReproduce.PDF.(ω₁,fermidirac))

    ω₁_effective = ω₁_effective_range(fermidirac, ω)

    #integral = quadgk(kernel,ω₁_effective[1], ω₁_effective[2]; rtol=1e-6)[1]
    integral = quadgk(kernel, -Inf, Inf; rtol=1e-6)[1]

    return - integral ./ (ω * 4π)

end


function Lambda(energy_vec::AbstractVector, adsorbate_m::WideBandLimitModelNDOF, configuration::SVector, temperature::Real, fermi_level::Real=0.0)

    # 1. Check available worker processes
    avail_workers = workers()
    has_workers = length(avail_workers) > 1 || (length(avail_workers) == 1 && avail_workers[1] != 1)
    n_workers = length(avail_workers)

    if has_workers
        # ----------------------------------------------------
        # 🚀 STRATEGY 1: MULTIPROCESSING (Hybrid Outer/Inner)
        # ----------------------------------------------------
        threads_per_worker = Threads.nthreads()
        @info "Hybrid Parallelism: Distributing $(length(energy_vec)) energies across $n_workers worker processes, with $threads_per_worker threads per worker."

        worker_func(e) = Lambda(e, adsorbate_m, configuration, temperature, fermi_level)
        results_iterator = pmap(worker_func, energy_vec)
        return collect(results_iterator)  # Vector{Matrix{Float64}}

    else
        # -----------------------------------------------------------------
        # ⚙️ STRATEGY 2: FALLBACK TO LOCAL MULTITHREADING
        # -----------------------------------------------------------------
        n_energy = length(energy_vec)

        @warn "No multiprocessing detected. Falling back to local multithreading ($(Threads.nthreads()) threads)."

        # Infer return type (Matrix{Float64}) from first call
        R = typeof(Lambda(energy_vec[1], adsorbate_m, configuration, temperature, fermi_level))
        Lambda_mat_vec = Vector{R}(undef, n_energy)

        @info "Starting multithreaded Λ(ω) matrix calculation..."
        Threads.@threads for i in 1:n_energy
            Lambda_mat_vec[i] = Lambda(energy_vec[i], adsorbate_m, configuration, temperature, fermi_level)
        end
        @info "Λ(ω) matrix calculation completed."

        return Lambda_mat_vec  # Vector{Matrix{Float64}}
    end
end


function Lambda(energy_vec::AbstractVector, adsorbate_m::AndersonImpurityModel1DOF, position::Real ,temperature::Real, fermi_level::Real=0.0)
    
    # 1. Check available worker processes
    avail_workers = workers()
    # A robust check: if there is more than just the main process (PID 1)
    has_workers = length(avail_workers) > 1 || (length(avail_workers) == 1 && avail_workers[1] != 1)
    n_workers = length(avail_workers)

    if has_workers
        # ----------------------------------------------------
        # 🚀 STRATEGY 1: MULTIPROCESSING (Hybrid Outer/Inner)
        # ----------------------------------------------------

        # Get the number of threads available to each worker (which is the same as the Master's setting)
        threads_per_worker = Threads.nthreads()
        
        @info "Hybrid Parallelism: Distributing $(length(energy_vec)) energies across $n_workers worker processes, with $threads_per_worker threads per worker." 

        # Define the work function. This closure captures all arguments (bath, model, etc.)
        # and transfers them once per worker via pmap's internal mechanism.
        worker_func(e) = Lambda(e, adsorbate_m, position, temperature, fermi_level)

        # pmap distributes the work element-by-element. Each element executes Lambda(e),
        # where cauchy_integral uses the worker's internal threads.

        results_iterator = pmap(worker_func, energy_vec)
        
        return collect(results_iterator) # Collect the results into a single Vector

    else
        # -----------------------------------------------------------------
        # ⚙️ STRATEGY 2: FALLBACK TO LOCAL MULTITHREADING (Original optimized code)
        # -----------------------------------------------------------------
        n_energy = length(energy_vec)

        @warn "No multiprocessing detected. Falling back to local multithreading ($(Threads.nthreads()) threads)."

        # Infer return type from first call, then fill all n_energy points in parallel
        R = typeof(Lambda(energy_vec[1], adsorbate_m, position, temperature, fermi_level))
        Lambda_au_vec = Vector{R}(undef, n_energy)

        @info "Starting multithreaded Λ(ω) calculation..."
        Threads.@threads for i in 1:n_energy
            Lambda_au_vec[i] = Lambda(energy_vec[i], adsorbate_m, position, temperature, fermi_level)
        end
        @info "Λ(ω) calculation completed."

        return Lambda_au_vec
    end
end



end