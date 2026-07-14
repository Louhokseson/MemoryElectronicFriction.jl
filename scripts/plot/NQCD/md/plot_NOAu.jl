using DrWatson
@quickactivate "MemoryElectronicFriction"

using CairoMakie
using Unitful, UnitfulAtomic
using HDF5
using HokseonPlots
using HokseonAssistant

colormap = HokseonPlots.NICECOLORS
HokseonAssistant.julia_build_procs()

if !isdefined(Main, :MemoryElectronicFriction)
    include(srcdir("MemoryElectronicFriction.jl"))
    using .MemoryElectronicFriction
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
# Config — must match run_md.jl's all_params_NOAu
# -----------------------------------------------------------------------------

all_params_NOAu = Dict{String, Any}(
    "mass"                  => [(14.007 * 15.999 / (14.007 + 15.999)) * u"u"],   # μ_NO — POGO is 1-atom
#    "Γ"                     => [1.5u"eV"], ## constant 1.5 eV
    "r0"                    => [[1.15u"Å", 5.0u"Å"]],    # (r, z); r0[1] is the frozen bond length when vibrational_state=nothing
    "translational_kinetic" => [1.0u"eV"],
    "state"                 => [1],
    "tmax"                  => [500.0u"fs"],
    "dt"                    => [0.25u"fs"],
    "termination_min_time"  => [10.0u"fs"],
    "termination_coord_idx" => [2],                       # check z
    "termination_threshold" => [5.0u"Å"],                 # scattered threshold
    # nothing → frozen bond (old behaviour). Integer ν → EBK-sample (r, ṙ)
    # at quantum number ν; bump trajectories to ~1000 for a ν ensemble.
    "vibrational_state"     => [0],                 # try [nothing, 0, 3, 16]
    "trajectories"          => [1000],
)
params_list_NOAu = dict_list(all_params_NOAu)

# -----------------------------------------------------------------------------
# Plot: surface and translational energy vs time for the NOAu BO run.
# OutputPosition is a 3D array of shape (ndofs, natoms, ntime) = (1, 1, N);
# slice [1,1,:] to get the bond-length time series.
# -----------------------------------------------------------------------------

function plot_NOAu(params)
    savingpath, savingname = dict_to_data_savename(params, "NOAu")
    full_data_path = datadir(savingpath, savingname)
    @info "Loading" full_data_path
    traj = load_trajectory(full_data_path)

    t_fs = ustrip.(auconvert.(u"fs", traj["Time"]))
    pos  = traj["OutputPosition"]
    r_A  = ustrip.(auconvert.(u"Å",  pos[1, 1, :]))
    z_A  = ustrip.(auconvert.(u"Å",  pos[2, 1, :]))
    E_eV = ustrip.(auconvert.(u"eV", traj["OutputKineticEnergy"]))

    # 1. Adjust figure size for a single, taller panel
    fig = Figure(size = (HokseonPlots.RESOLUTION[1]*4, HokseonPlots.RESOLUTION[2]*3),
                 figure_padding = (1, 1, 1, 5),
                 fonts = (;regular = projectdir("fonts", "MinionPro-Capt.otf")))

    # 2. Left Y-Axis (Position)
    ax1 = Axis(fig[1, 1]; 
               xlabel = "t / fs", 
               ylabel = "Coordinate / Å",
               xgridvisible = false, # Removes vertical grid lines
               ygridvisible = false  # Removes horizontal grid lines
               ) # 3. Removed Title

    ax2 = Axis(fig[1, 2];
               xlabel = "t / fs", 
               yaxisposition = :right, 
               ylabel = "Kinetic Energy / eV",
               ylabelcolor = :blue, # Optional: color-code label
               yticklabelcolor = :blue,
               xgridvisible = false, # Removes vertical grid lines
               ygridvisible = false  # Removes horizontal grid lines
               )

    #hidexdecorations!(ax2) 

    # Plot the data
    lines!(ax1, t_fs, r_A; linewidth = 3, color = colormap[2], label = "r (N–O)")
    lines!(ax1, t_fs, z_A; linewidth = 3, color = colormap[4], label = "z surface distance")
    lines!(ax2, t_fs, E_eV; linewidth = 3, color = :blue)

    # Link the x-axes so zooming/panning stays synchronized
    #linkxaxes!(ax1, ax2)

    Label(fig[1,1], "BO-MD\nNOAu\nΓ = 1.5 eV\n ν = $(get(params, "vibrational_state", nothing))"; tellwidth=false, tellheight=false, valign=:center, halign=:right, padding=(10,10,10,10),fontsize=16)

    Legend(fig[1,1], ax1; tellwidth=false, tellheight=false, valign=:top, halign=:center, margin=(5,5,5,5))

    return fig
end

for params in params_list_NOAu
    fig = plot_NOAu(params)
    #save(projectdir("plots", "md", "NOAu_ν=$(get(params, "vibrational_state", "frozen")).pdf"), fig)
    display(fig)
end
