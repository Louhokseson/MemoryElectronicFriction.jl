using DrWatson
@quickactivate "HokseonReproduce"

using QuadGK

# Your original function
g(x, y) = 1/(x - 1) + 1/(x + 1) + 1/(y - 1) + 1/(y + 1)

# The function to integrate
f(x1, x) = g(x1, x + x1)

# Find singularities in x₁ domain
function singularities(x)
    return sort([
        -1.0, 1.0,           # singularities from x₁
        -1.0 - x, 1.0 - x    # singularities from x + x₁
    ])
end

function interval_limits(a::Real, b::Real, singularities, ϵ)
    """
    a: Lower limit of the interval
    b: Upper limit of the interval
    singularities: Vector of singularity points
    """
    a_new = a
    b_new = b
    if a in singularities
        a_new = a + ϵ  # Shift slightly to avoid singularity
    end
    if b in singularities
        b_new = b - ϵ  # Shift slightly to avoid singularity
    end
    return a_new, b_new
end

# Cauchy principal value integration
function principal_value_integral(f, x::Real; ε=1e-3)
    # Get all singularities in x₁
    sing_pts = singularities(x)

    # Integration intervals: split at each singularity
    bounds = [-Inf; sing_pts; Inf]

    total = 0.0
    total_error = 0.0

    for i in 1:length(bounds)-1
        a, b = interval_limits(bounds[i], bounds[i+1], sing_pts, ε)

        integral, error = quadgk(x1 -> f(x1, x), a, b; rtol=1e-6)
        total += integral
        total_error += error
    end

    return total, total_error
end

# 🔍 Example
val, err = principal_value_integral(f, 1.5; ε=2e-3)
println("Principal value of the integral: $val")
println("Estimated numerical error: $err")


