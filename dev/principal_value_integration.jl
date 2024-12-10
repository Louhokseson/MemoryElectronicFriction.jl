using DrWatson
@quickactivate "HokseonReproduce"

using QuadGK

function principal_value_integral(f, a, b, singularity)
    if !(a < singularity && singularity < b)
        error("Singularity must be between a and b")
    end
    left_integral = quadgk(f, a, singularity; atol=1e-8, rtol=1e-8)[1]
    right_integral = quadgk(f, singularity, b; atol=1e-8, rtol=1e-8)[1]
    return left_integral + right_integral
end

a, b = 0.0, 1.0
singularity = 0.5
result = principal_value_integral(x -> 1/x, a, b, singularity)
println("Principal value integral of 1/x from $a to $b: $result")