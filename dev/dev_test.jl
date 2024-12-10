using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce: TrapezoidalRule, ShenviGaussLegendre
end

include("dev_parameters.jl")
@unpack discretization, centre, width, nstates = bath_params

bandmin = centre - width/2
bandmax = centre + width/2

bath = eval(discretization)(nstates, bandmin, bandmax)
