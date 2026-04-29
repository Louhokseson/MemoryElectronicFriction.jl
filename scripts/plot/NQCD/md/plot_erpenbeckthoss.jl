using DrWatson
@quickactivate "HokseonReproduce"

using CairoMakie
using Unitful, UnitfulAtomic
using HDF5
using HokseonPlots
using HokseonAssistant

colormap = HokseonPlots.NICECOLORS
HokseonAssistant.julia_build_procs()

# -----------------------------------------------------------------------------
# Savename helpers (mirror run_md.jl so we can locate the .h5 for a given
# parameter dict). Kept inline rather than shared to keep this plot script
# self-contained.
# -----------------------------------------------------------------------------

_savename_value(v::Unitful.Quantity) = ustrip(v)
_savename_value(v::AbstractVector{<:Unitful.Quantity}) =
    length(v) == 1 ? ustrip(v[1]) : join(string.(ustrip.(v)), "-")
_savename_value(v::AbstractVector) = length(v) == 1 ? v[1] : v
_savename_value(v) = v

function _sanitize_for_savename(param_dict::Dict{String,Any})
    Dict{String,Any}(k => _savename_value(v) for (k, v) in param_dict)
end

function dict_to_data_savename(param_dict::Dict{String,Any}, model_folder::AbstractString)
    savingpath = joinpath("sims", "md", model_folder)
    savingname = savename(_sanitize_for_savename(param_dict), "h5")
    return (savingpath, savingname)
end

# -----------------------------------------------------------------------------
# Load a single trajectory from the FileReduction h5 layout: /<traj_idx>/<qty>
# For `trajectories = 1` in run_md.jl, there's one top-level group; pull it.
# -----------------------------------------------------------------------------

function load_trajectory(full_data_path)
    h5open(full_data_path, "r") do fid
        first_key = first(keys(fid))
        group = fid[first_key]
        return Dict(q => Array(group[q]) for q in keys(group))
    end
end

# -----------------------------------------------------------------------------
# Config — must match run_md.jl's all_params_et
# -----------------------------------------------------------------------------

all_params_et = Dict{String, Any}(
    "mass"                  => [10.54u"u"],
    "Γ"                     => [0.25u"eV"],
    "r0"                    => [[5.0u"Å"]],              # 1 DOF: surface distance
    "translational_kinetic" => [3.0u"eV"],
    "state"                 => [1],
    "tmax"                  => [200.0u"fs"],
    "dt"                    => [0.01u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [1],                       # check r
    "termination_threshold" => [5.0u"Å"],                 # dissociation threshold
)
params_list_et = dict_list(all_params_et)

# -----------------------------------------------------------------------------
# Plot: surface and translational energy vs time for the ErpenbeckThoss BO run.
# OutputPosition is a 3D array of shape (ndofs, natoms, ntime) = (1, 1, N);
# slice [1,1,:] to get the bond-length time series.
# -----------------------------------------------------------------------------

function plot_erpenbeckthoss(params)
    savingpath, savingname = dict_to_data_savename(params, "ErpenbeckThoss")
    full_data_path = datadir(savingpath, savingname)
    @info "Loading" full_data_path
    traj = load_trajectory(full_data_path)

    t_fs = ustrip.(auconvert.(u"fs", traj["Time"]))
    pos  = traj["OutputPosition"]
    r_A  = ustrip.(auconvert.(u"Å",  pos[1, 1, :]))
    E_eV = ustrip.(auconvert.(u"eV", traj["OutputKineticEnergy"]))

    # 1. Adjust figure size for a single, taller panel
    fig = Figure(size = (HokseonPlots.RESOLUTION[1]*2, HokseonPlots.RESOLUTION[2]*3),
                 figure_padding = (1, 1, 1, 5),
                 fonts = (;regular = projectdir("fonts", "MinionPro-Capt.otf")))

    # 2. Left Y-Axis (Position)
    ax1 = Axis(fig[1, 1]; 
               xlabel = "t / fs", 
               ylabel = "Surface Distance / Å",
               xgridvisible = false, # Removes vertical grid lines
               ygridvisible = false  # Removes horizontal grid lines
               ) # 3. Removed Title

    ax2 = Axis(fig[1, 1]; 
               yaxisposition = :right, 
               ylabel = "Kinetic Energy / eV",
               ylabelcolor = :red, # Optional: color-code label
               yticklabelcolor = :red,
               xgridvisible = false, # Removes vertical grid lines
               ygridvisible = false  # Removes horizontal grid lines
               )

    hidexdecorations!(ax2) 

    # Plot the data
    lines!(ax1, t_fs, r_A; linewidth = 3, color = :black)
    lines!(ax2, t_fs, E_eV; linewidth = 3, color = :red)

    # Link the x-axes so zooming/panning stays synchronized
    linkxaxes!(ax1, ax2)

    Label(fig[1,1], "BO-MD\nErpenbeckThoss\nΓ = $(ustrip(params["Γ"])) eV"; tellwidth=false, tellheight=false, valign=:bottom, halign=:right, padding=(10,10,10,10),fontsize=16)

    return fig
end

fig = plot_erpenbeckthoss(params_list_et[1])
#save(projectdir("plots", "md", "erpenbeckthoss.pdf"), fig)
display(fig)
