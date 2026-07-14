using Distributed
using DrWatson
@quickactivate "MemoryElectronicFriction"

using HDF5
using DelimitedFiles
using ClusterManagers
using HokseonAssistant
HokseonAssistant.julia_build_procs()


# Activate project everywhere
@everywhere using DrWatson
@everywhere @quickactivate "MemoryElectronicFriction"
@everywhere using MemoryElectronicFriction
@everywhere using Unitful, UnitfulAtomic
@everywhere using NQCModels.QuantumModels
@everywhere using NQCModels