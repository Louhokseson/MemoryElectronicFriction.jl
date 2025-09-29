# HokseonReproduce

![CI Workflow Status](https://github.com/Louhokseson/HokseonReproduce/actions/workflows/CI.yml/badge.svg)

This repo Requires Julia version `>=1.11`. You can find the latest stable version from [here](https://julialang.org/downloads/).
> HokseonReproduce

It is authored by Hokseon.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "HokseonReproduce"
```
which auto-activate the project and enable local path handling from DrWatson.

### Frequency Dependent Friction $\Lambda(\omega)$
The repo is initially developed for replicating the frequency dependent electronic friction from a  adsorbate coupling a electronic bath provided  by [Brandbyge & his collabrators](https://doi.org/10.1103/PhysRevB.52.6042). 

#### Implementation 
`FrequencyLambda.Lambda()` conducts a serie of computationally static steps involving integration with respect to the outcome from matrix products. To broadcast the frequency dependent friction with an array of $\omega$ s which apparantly shares the same operational memory, it is adviced to apply [**Multithreading**](https://docs.julialang.org/en/v1/manual/multi-threading/) in the Julia process. Simply by
```bash
julia -t auto your_script_to_call_Lambda.jl
```
the calculation speed can be significantly ramped up based on the number of available threads in your machine.
