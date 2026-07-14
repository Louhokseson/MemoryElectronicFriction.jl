"""

NOAuAdsorbate is a model for NO molecule scattering on Au(111) surface.

This model is a 2-D model with bond length and distance to surface as two dimensions.

J. Phys. Chem. C 2023, 127, 15257−15270 https://doi.org/10.1021/acs.jpcc.3c03591

This model is based on wideband limit approximation.

r : N–O bond length
z : molecule–surface distance

"""

#parameters = [1.9535329365277612, -0.26876384605775233, 6.571336959902485, 2.519431394450958, 1.295009818269084, 4.152792292961294, 1.0015030848580035, 1.2350312403781256, 2.417096730391661, 8.958749759728342]


@kwdef struct NOAuAdsorbate <: WideBandLimitModelNDOF

    ndof::Int = 2

    ## coupling
    Γ::AbstractFloat = austrip(1.5u"eV")
    Vₖ::AbstractFloat = sqrt(Γ/2π)
    ã::AbstractFloat = austrip(10u"Å")


    ## U₀ U₁ parameters
    r₀::AbstractFloat = austrip(1.1510u"Å")
    a₀::AbstractFloat = austrip(2.7968u"Å^-1")
    D₀::AbstractFloat = austrip(6.610u"eV")
    b₀::AbstractFloat = austrip(1.9535329365277612u"Å^-1")
    z₀::AbstractFloat = austrip(-0.26876384605775233u"Å")
    c₀::AbstractFloat = austrip(6.571336959902485u"eV")
    a₁::AbstractFloat = austrip(2.519431394450958u"Å^-1")
    r₁::AbstractFloat = austrip(1.295009818269084u"Å")
    D₁::AbstractFloat = austrip(4.152792292961294u"eV")
    a₂::AbstractFloat = austrip(1.0015030848580035u"Å^-1")
    z₁::AbstractFloat = austrip(1.2350312403781256u"Å")
    D₂::AbstractFloat = austrip(2.417096730391661u"eV")
    c₁::AbstractFloat = austrip(8.958749759728342u"eV")
end

function V_k(z::Real, adsorbate_m::NOAuAdsorbate)
    ã, Vₖ = getfield.(Ref(adsorbate_m), (:ã, :Vₖ))
    return Vₖ * (1 - tanh(z / ã))
end

function dV_k_dz(z::Real, adsorbate_m::NOAuAdsorbate)
    ã, Vₖ = getfield.(Ref(adsorbate_m), (:ã, :Vₖ))
    return - Vₖ / ã * sech(z / ã)^2
end

function MorsePotential(x, D, a)
    return D * (exp(-2 * a * x) - 2 * exp(-1 * a * x))
end

function U₀(r::Real, z::Real, adsorbate_m::NOAuAdsorbate)
    r₀, a₀, D₀, b₀, z₀, c₀ = getfield.(Ref(adsorbate_m), (:r₀, :a₀, :D₀, :b₀, :z₀, :c₀))
    return MorsePotential(r - r₀, D₀, a₀) + exp(-b₀ * (z - z₀)) + c₀
end

function U₁(r::Real, z::Real, adsorbate_m::NOAuAdsorbate)
    a₁, r₁, D₁, a₂, z₁, D₂, c₁ = getfield.(Ref(adsorbate_m), (:a₁, :r₁, :D₁, :a₂, :z₁, :D₂, :c₁))
    return MorsePotential(r - r₁, D₁, a₁) + MorsePotential(z - z₁, D₂, a₂) + c₁
end


function MemoryElectronicFriction.Δ(z::Real, adsorbate_m::NOAuAdsorbate)
    Vk = V_k(z, adsorbate_m)
    return π * Vk^2
end


function MemoryElectronicFriction.adsorbate_h(r::Real, z::Real, adsorbate_m::NOAuAdsorbate)
    U₀_val = U₀(r, z, adsorbate_m)
    U₁_val = U₁(r, z, adsorbate_m)
    return U₁_val - U₀_val
end

"""
    dh_dx(r, z, adsorbate_m::NOAuAdsorbate) -> SVector{2}
    dh_dx(rz::SVector{2}, adsorbate_m::NOAuAdsorbate) -> SVector{2}

Gradient of the diabatic energy gap `h = U₁ - U₀` with respect to the two
degrees of freedom `x = (r, z)`, where `r` is the N–O bond length and `z` is
the molecule–surface distance.

Returns `SA[∂h/∂r, ∂h/∂z]`.

## Partial derivatives

**∂h/∂r** — only the Morse terms in `r` contribute:

```
∂U₁/∂r = 2 a₁ D₁ [exp(-a₁(r-r₁)) - exp(-2a₁(r-r₁))]
∂U₀/∂r = 2 a₀ D₀ [exp(-a₀(r-r₀)) - exp(-2a₀(r-r₀))]
∂h/∂r  = ∂U₁/∂r - ∂U₀/∂r
```

**∂h/∂z** — `U₁` has a Morse term in `z`; `U₀` has an exponential repulsion:

```
∂U₁/∂z = 2 a₂ D₂ [exp(-a₂(z-z₁)) - exp(-2a₂(z-z₁))]
∂U₀/∂z = -b₀ exp(-b₀(z-z₀))
∂h/∂z  = ∂U₁/∂z - ∂U₀/∂z
```
"""
function MemoryElectronicFriction.dh_dx(r::Real, z::Real, adsorbate_m::NOAuAdsorbate)
    a₁, r₁, D₁, a₂, z₁, D₂ = getfield.(Ref(adsorbate_m), (:a₁, :r₁, :D₁, :a₂, :z₁, :D₂))
    a₀, r₀, D₀, b₀, z₀ = getfield.(Ref(adsorbate_m), (:a₀, :r₀, :D₀, :b₀, :z₀))
    # d/dr: only Morse(r - rᵢ) terms contribute
    dU₁_dr = 2 * a₁ * D₁ * (exp(-a₁ * (r - r₁)) - exp(-2 * a₁ * (r - r₁)))
    dU₀_dr = 2 * a₀ * D₀ * (exp(-a₀ * (r - r₀)) - exp(-2 * a₀ * (r - r₀)))
    # d/dz: Morse(z - z₁) for U₁, exp(-b₀*(z - z₀)) for U₀
    dU₁_dz = 2 * a₂ * D₂ * (exp(-a₂ * (z - z₁)) - exp(-2 * a₂ * (z - z₁)))
    dU₀_dz = -b₀ * exp(-b₀ * (z - z₀))
    return SA[dU₁_dr - dU₀_dr, dU₁_dz - dU₀_dz]
end

"""
    dΔ_dx(r, z, adsorbate_m::NOAuAdsorbate) -> SVector{2}
    dΔ_dx(rz::SVector{2}, adsorbate_m::NOAuAdsorbate) -> SVector{2}

Gradient of the hybridisation width `Δ(z) = π Vₖ(z)²` with respect to the
two degrees of freedom `x = (r, z)`.

Returns `SA[∂Δ/∂r, ∂Δ/∂z]`.

## Partial derivatives

`Δ` depends only on `z` through the coupling `Vₖ(z) = Vₖ (1 - tanh(z/ã))`,
so the `r` component is identically zero:

```
∂Δ/∂r = 0
∂Δ/∂z = 2π Vₖ(z) · dVₖ/dz,   dVₖ/dz = -Vₖ/ã · sech²(z/ã)
```
"""
function MemoryElectronicFriction.dΔ_dx(r::Real, z::Real, adsorbate_m::NOAuAdsorbate)
    # Δ = π * V_k(z)² depends only on z, so dΔ/dr = 0
    # dΔ/dz = 2π * V_k * dV_k/dz
    Vk  = V_k(z, adsorbate_m)
    dVk = dV_k_dz(z, adsorbate_m)
    return SA[zero(r), 2 * π * Vk * dVk]
end

# SVector{2} overloads — rz = SA[r, z]
U₀(rz::SVector{2}, m::NOAuAdsorbate)                              = U₀(rz[1], rz[2], m)
U₁(rz::SVector{2}, m::NOAuAdsorbate)                              = U₁(rz[1], rz[2], m)
MemoryElectronicFriction.adsorbate_h(rz::SVector{2}, m::NOAuAdsorbate)    = MemoryElectronicFriction.adsorbate_h(rz[1], rz[2], m)
MemoryElectronicFriction.Δ(rz::SVector{2}, m::NOAuAdsorbate)              = MemoryElectronicFriction.Δ(rz[2], m)
MemoryElectronicFriction.dh_dx(rz::SVector{2}, m::NOAuAdsorbate)          = MemoryElectronicFriction.dh_dx(rz[1], rz[2], m)
MemoryElectronicFriction.dΔ_dx(rz::SVector{2}, m::NOAuAdsorbate)          = MemoryElectronicFriction.dΔ_dx(rz[1], rz[2], m)