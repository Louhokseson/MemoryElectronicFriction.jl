using Distributed
using DrWatson
@quickactivate "HokseonReproduce"

using HDF5
using DelimitedFiles
using ClusterManagers
using HokseonAssistant
HokseonAssistant.julia_build_procs()


# Activate project everywhere
@everywhere using DrWatson
@everywhere @quickactivate "HokseonReproduce"
@everywhere using HokseonReproduce
@everywhere using Unitful, UnitfulAtomic
@everywhere using NQCModels.QuantumModels
@everywhere using NQCModels