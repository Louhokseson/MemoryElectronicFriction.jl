"""
    POGOModel

NQCModels-compatible 2D diabatic model for NO scattering on Au(111),
adapted from https://github.com/NQCD/SurfaceScatteringMQC.

Implements the NQCModels QuantumModel interface so that
`NQCCalculators.evaluate_friction` can be used directly.

Degrees of freedom:
  r : N–O bond length   (index 1)
  z : molecule–surface distance (index 2)

Same physical model and parameters as NOAuAdsorbate — use POGOModel
when you need the NQCModels interface (e.g. Markovian MDEF via NQCDynamics),
and NOAuAdsorbate when you need the HokseonReproduce friction interface.
"""

# --- primitive potential types ---

struct MorsePot{T}
    a::T
    x0::T
    D::T
end

function (m::MorsePot)(x)
    (; a, x0, D) = m
    return D * (exp(-2a * (x - x0)) - 2exp(-a * (x - x0)))
end

function ∂(m::MorsePot, x)
    (; a, x0, D) = m
    return D * (-2a * exp(-2a * (x - x0)) + 2a * exp(-a * (x - x0)))
end

# Predefined Morse for the neutral (U₀ r-component) — MorseNO parameters from NOAuAdsorbate
MorseNOPot()  = MorsePot(austrip(2.7968u"Å^-1"), austrip(1.1510u"Å"), austrip(6.610u"eV"))

struct RepelPot{T}
    b::T
    x0::T
end

function (r::RepelPot)(x)
    (; b, x0) = r
    return exp(-b * (x - x0))
end

function ∂(r::RepelPot, x)
    (; b, x0) = r
    return -b * exp(-b * (x - x0))
end

struct TwoDPot{X,Y,T}
    x_component::X
    y_component::Y
    shift::T
end

function (p::TwoDPot)(x, y)
    (; x_component, y_component, shift) = p
    return x_component(x) + y_component(y) + shift
end

function ∂(p::TwoDPot, x, y)
    (; x_component, y_component) = p
    return SVector{2}(∂(x_component, x), ∂(y_component, y))
end

struct SmoothStepPot{T}
    V̄ₖ::T
    x̃::T
    ã::T
end

function (s::SmoothStepPot)(x)
    (; V̄ₖ, x̃, ã) = s
    return V̄ₖ * (1 - tanh((x - x̃) / ã))
end

function ∂(s::SmoothStepPot, x)
    (; V̄ₖ, x̃, ã) = s
    return -V̄ₖ * sech((x - x̃) / ã)^2 / ã
end

# --- main model struct ---

struct POGOModel{T} <: NQCModels.QuantumModels.QuantumFrictionModel
    U0::TwoDPot{MorsePot{T}, RepelPot{T}, T}
    U1::TwoDPot{MorsePot{T}, MorsePot{T}, T}
    Vk::SmoothStepPot{T}
end

"""
    POGOModel(parameters; Γ = austrip(1.5u"eV"))

Construct a POGOModel from the 10-element parameter array matching NOAuAdsorbate:
`[b₀, z₀, c₀, a₁, r₁, D₁, a₂, z₁, D₂, c₁]`
"""
function POGOModel(parameters::AbstractArray; Γ = austrip(1.5u"eV"))
    U0 = TwoDPot(
        MorseNOPot(),
        RepelPot(
            austrip(parameters[1] * u"Å^-1"),
            austrip(parameters[2] * u"Å")
        ),
        austrip(parameters[3] * u"eV")
    )

    U1 = TwoDPot(
        MorsePot(
            austrip(parameters[4] * u"Å^-1"),
            austrip(parameters[5] * u"Å"),
            austrip(parameters[6] * u"eV")
        ),
        MorsePot(
            austrip(parameters[7] * u"Å^-1"),
            austrip(parameters[8] * u"Å"),
            austrip(parameters[9] * u"eV")
        ),
        austrip(parameters[10] * u"eV")
    )

    V̄ₖ = sqrt(Γ / 2π)
    Vk  = SmoothStepPot(V̄ₖ, austrip(0.0u"Å"), austrip(10.0u"Å"))

    return POGOModel(U0, U1, Vk)
end

"""
    POGOModel(; Γ = austrip(1.5u"eV"))

Construct with the default NOAuAdsorbate parameters.
"""
function POGOModel(; Γ = austrip(1.5u"eV"))
    parameters = [1.9535329365277612, -0.26876384605775233, 6.571336959902485,
                  2.519431394450958,   1.295009818269084,   4.152792292961294,
                  1.0015030848580035,  1.2350312403781256,  2.417096730391661,
                  8.958749759728342]
    return POGOModel(parameters; Γ)
end

# --- NQCModels interface ---

NQCModels.nstates(::POGOModel) = 2
NQCModels.ndofs(::POGOModel)   = 2   # r (bond length) and z (surface distance)

# R is a (ndofs × natoms) matrix — for a single particle: 2×1, R[1,1]=r, R[2,1]=z
function NQCModels.potential!(model::POGOModel, V::Hermitian, R::AbstractMatrix)
    (; U0, U1, Vk) = model
    x, y = R[1, 1], R[2, 1]
    V11 = U0(x, y)
    V22 = U1(x, y)
    V12 = Vk(y)
    V.data .= SMatrix{2,2}(V11, V12, V12, V22)
    return V
end

# D is a (ndofs × natoms) matrix of Hermitian matrices — D[1,1]=∂/∂r, D[2,1]=∂/∂z
function NQCModels.derivative!(model::POGOModel, D::AbstractMatrix{<:Hermitian}, R::AbstractMatrix)
    (; U0, U1, Vk) = model
    x, y = R[1, 1], R[2, 1]
    ∂U0 = ∂(U0, x, y)
    ∂U1 = ∂(U1, x, y)
    ∂Vk = ∂(Vk, y)
    D[1, 1] = Hermitian(SMatrix{2,2}(∂U0[1], 0.0,  0.0,  ∂U1[1]))  # ∂/∂r
    D[2, 1] = Hermitian(SMatrix{2,2}(∂U0[2], ∂Vk,  ∂Vk,  ∂U1[2])) # ∂/∂z
    return D
end
