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

## NO/Au(111) scattering — the 2-D Gardner–Habershon–Maurer model

A reduced model for vibrationally inelastic scattering of NO from Au(111): a
two-state (neutral / charge-transfer) Newns–Anderson Hamiltonian whose **ground
adiabatic surface** depends on two coordinates — the N–O bond length `r` and the
molecule–surface distance `z` — which we propagate with Born–Oppenheimer MD.
Implementation: [`POGOModel`](src/AndersonImpurityModels/adsorbates/POGOModel.jl)
(NQCModels interface); BO-MD driver in [`run_md.jl`](scripts/compute/NQCD/run_md.jl);
adapted from [NQCD/SurfaceScatteringMQC](https://github.com/NQCD/SurfaceScatteringMQC).

**Initial NO vibration (EBK).** The incoming NO is prepared in vibrational state
`ν` by Einstein–Brillouin–Keller (EBK) quantisation of the 1-D bond, then
phase-space sampled. The animation walks through the logic for `ν = 16`:

1. **Quantise** — freeze the molecule far from the surface (`z = 10 Å`) to get the
   1-D bond binding curve `V(r)`, and raise the energy until the EBK action
   `∮ p dr = 2π(ν+½)ℏ` hits the integer `ν`. That fixes the bond energy `Eν`.
2. **Sample** — at fixed `Eν` the bond is a 1-D oscillator; draw `(r, ṙ)` snapshots
   uniformly in time along the orbit (`½μṙ² = Eν − V(r)`). Positions pile up at the
   turning points; the kinetic energy spans `0 → Eν − V_min`.

<p align="center">
  <img src="plots/ebk_sampling/ebk_sampling_nu16.gif" width="900"
       alt="EBK initial vibrational sampling for ν = 16"/>
  <br/>
  <em>EBK preparation of the NO(ν = 16) initial state. <b>Left:</b> the 1-D bond
  binding curve and its quantised levels — the energy is raised until the EBK action
  equals the integer ν, fixing E<sub>ν</sub>. <b>Centre:</b> the classical orbit at
  E<sub>ν</sub> in phase space (r, ṙ) with sampled snapshots. <b>Right:</b> the resulting
  distributions of bond length (top) and vibrational kinetic energy (bottom).</em>
</p>

The final vibrational state of the scattered trajectories is read off the same
binding curve — see [`run_vib_state_noau.jl`](scripts/compute/cpa/NOAu/run_vib_state_noau.jl).
Reproduce the animation with [`dev/CPA/animate_ebk_sampling.jl`](dev/CPA/animate_ebk_sampling.jl).

## Frequency Dependent Friction $\Lambda(\omega)$
The repo is initially developed for replicating the frequency dependent electronic friction from a  adsorbate coupling a electronic bath provided  by [Brandbyge & his collabrators](https://doi.org/10.1103/PhysRevB.52.6042). 

### Implementation 
`FrequencyLambda.Lambda()` conducts a serie of computationally static steps involving integration with respect to the outcome from matrix products. To broadcast the frequency dependent friction with an array of $\omega$ s which apparantly shares the same operational memory, it is adviced to apply [**Multithreading**](https://docs.julialang.org/en/v1/manual/multi-threading/) in the Julia process. Simply by
```bash
julia -t auto your_script_to_call_Lambda.jl
```
the calculation speed can be significantly ramped up based on the number of available threads in your machine.
