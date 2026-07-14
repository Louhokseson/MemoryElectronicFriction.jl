## Only for the main process
using Distributed
using DrWatson
@quickactivate "MemoryElectronicFriction" ## Activate project everywhere
import Pkg; Pkg.precompile() ## Precompile packages in master to speed up workers' precompilation
using HDF5
using HokseonAssistant
HokseonAssistant.julia_build_procs()


# Load packages everywhere
@everywhere using MemoryElectronicFriction
@everywhere using Unitful, UnitfulAtomic
@everywhere using NQCModels.QuantumModels
@everywhere using NQCModels


# For a single position, return the full Λ(ω) sweep in atomic units.
# Worker-side construction of the adsorbate model avoids shipping it across procs.
@everywhere function lambda_omega_at_position(position_Å::Float64,
                                              energy_eV::Vector{Float64},
                                              Γ_eV::Float64,
                                              T_K::Float64)
    adsorbate_m    = ErpenbeckThossAdsorbate(Γ = austrip(Γ_eV * u"eV"))
    energy_au      = austrip.(energy_eV .* u"eV")
    temperature_au = austrip(T_K * u"K")
    position_au    = austrip(position_Å * u"Å")
    return FrequencyLambda.Lambda.(energy_au, Ref(adsorbate_m),
                                   Ref(position_au), Ref(temperature_au))
end


# ---------------------------------------------------------------------------
# CLI argument parsing — accepts --gamma <float> and --temperature <int>.
# Falls back to the full sweep when run interactively (no args).
# ---------------------------------------------------------------------------

let i = 1, _gamma = nothing, _temp = nothing
    while i <= length(ARGS)
        if ARGS[i] == "--gamma" && i < length(ARGS)
            _gamma = parse(Float64, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--temperature" && i < length(ARGS)
            _temp  = parse(Float64, ARGS[i+1]); i += 2
        else
            i += 1
        end
    end
    global const CLI_GAMMA = _gamma
    global const CLI_TEMP  = _temp
end



all_params = Dict{String, Any}(
    "Gamma_eV" => CLI_GAMMA === nothing ? [0.02, 0.2, 0.8] : [CLI_GAMMA],
    "T_K"      => CLI_TEMP  === nothing ? [300.0] : [CLI_TEMP],
    "omega_eV" => [collect(1e-6:0.01:4.0)],                     # single array param (extra [])
    "x_Ang"    => [collect(range(1.85, 2.15, length = 1000))],  # single array param (extra [])
)
params_list = dict_list(all_params)

path = mkpath(datadir("sims", "friction", "ErpenbeckThoss"))

for (i, p) in enumerate(params_list)
    @unpack Gamma_eV, T_K, omega_eV, x_Ang = p

    ω_eV = omega_eV
    x_Å  = x_Ang

    # savename can't stringify the ω/x arrays in `p`, so name from scalars only.
    # Derive the ω step and x count from the arrays (not hardcoded) so the
    # filename always matches the actual grid and different resolutions don't
    # collide on the same name.
    domega_eV = length(ω_eV) > 1 ? ω_eV[2] - ω_eV[1] : 0.0
    Nx        = length(x_Å)
    name = savename("Lambda_contour",
                    (Gamma_eV = Gamma_eV, T_K = T_K, domega_eV = domega_eV, Nx = Nx),
                    "h5")
    full_data_path = joinpath(path, name)

    if isfile(full_data_path)
        @info "Skipping (already saved)" full_data_path
        continue
    end

    @info "Λ(ω, x) sweep $i/$(length(params_list))" Gamma_eV T_K N_ω = length(ω_eV) N_x = length(x_Å)

    # ---- Λ(ω, x) sweep, parallel over positions ----
    Λ_cols_au   = pmap(x -> lambda_omega_at_position(x, ω_eV, Gamma_eV, T_K), x_Å)
    Λ_matrix_au = reduce(hcat, Λ_cols_au)        # [N_ω, N_x], atomic units (me / au_t)

    # ---- Save raw grid (atomic units) + axes + metadata ----
    # Keep data/ canonical: store au values and the grid only. Unit conversion,
    # transpose, NaN-masking and color clipping all live in the plot script.
    h5open(full_data_path, "w") do fid
        fid["Lambda_au"] = Λ_matrix_au            # [N_ω, N_x] in atomic units
        fid["omega_eV"]  = ω_eV                   # length N_ω
        fid["x_Ang"]     = x_Å                    # length N_x
        attributes(fid)["Gamma_eV"] = Gamma_eV
        attributes(fid)["T_K"]      = T_K
        attributes(fid)["dims"]     = "Lambda_au[N_omega, N_x]; atomic units me/au_t"
        attributes(fid)["model"]    = "ErpenbeckThossAdsorbate"
    end

    @info "Saved Λ(ω, x) grid" full_data_path size = size(Λ_matrix_au)
end
