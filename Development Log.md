
### 2024/12/03

- [x] Try to review what I have done
- The implementation of the projected density of states
- Basic structure of the HokseonReproduce
- [x] Learn the way to turn HokseonReproduce into a package 
- Do something similar with NQCDModel
- [x] Turn HokseonReproduce into a semi-package by defining a lot of module
- [x] Create the type of model for HokseonReproduce
- [x] Call the DOS function by HokseonReproduce module
- (Top level function defined in HokseonReproduce.jl module)


### 2024/12/05
- [x] Build the module Distributions Lorentzian and FermiDirac
- [x] Coordinate module Distributions with Parent module HokseonReproduce

### 2024/12/09
- [x] Implemented the bath discretization from NQCModels.jl
- [ ] Try to do the subtype graph for the HokseonReproduce. 人都癫 I hate this crap.

### 2024/12/10
- [x] Implement the $\Gamma(\omega_1,\omega_2)$  on HokseonReproduce (Not done yet, but making progress on dev_Gamma.jl)
-  `bath.bathstates`  Assume our electronic distribution is following the Fermi Dirac distribution? 100 electron 200 states groundstate 
- Implement $V_{ak}(\omega)'$ onto HokseonReproduce
- [ ] If I have time, try to make the plot like alex did.
- Have a look into those different package in julia and compare to Alex's code.
- [x] Changed the name of Distributions into DistributionTools
- [x] Added a gaussian struct into DistributionTools

### 2024/12/11

- [ ] Have a look into the Cauchy principle value. Have a think about the implementation.
- [ ] Try to Implement the $\Gamma(\omega_1,\omega_2)$ The configuration index summation.
- Try to figure out concepts of $R_{ak}$ ?
Is it okay to have
$$
R_{ak} = R_{ka}^*?
$$

I assume it should be equal because the symmetry? 

Reason: The coupling terms in our Hamiltonian is identical if we consider them are real value.

In our case, it can boil down to the 
$$
G_{a}^{\text{ret}}(\omega)T_kG_{k}^{\text{ret}(0)}(\omega) ?= G_{k}^{\text{ret}}(\omega)T_aG_{a}^{\text{ret}(0)}(\omega)
$$
In our discussion above we should have 
$$
T_a = T_k
$$
Because both of them the coupling between impurity and bath. Should be identical. Now we need to figure out
$$
G_a^{ret} G_k^{ret(0)} = G_k^{ret}G^{ret(0)}_a ?
$$
- [x] $R_ak$ fully understood!!!!!!!!!! 

