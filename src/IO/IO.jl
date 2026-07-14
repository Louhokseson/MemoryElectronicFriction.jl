module IO

using ..MemoryElectronicFriction: MemoryElectronicFriction
using DrWatson: datadir, savename
using HDF5: h5open
using Unitful: Unitful, ustrip

export dict_to_data_savename, load_md_trajectories, ndofs,
       CPA_dict_to_data_savename

include("md_io.jl")
include("cpa_io.jl")

end
