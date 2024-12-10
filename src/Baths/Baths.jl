module Baths

using Reexport: @reexport
using ..HokseonReproduce: HokseonReproduce, HokseonReproduce.Bath
    

include("wideband/widebandbath.jl")
@reexport using .WideBand

"""
@reexport : import the module WideBand and export the functionalities of WideBand.
"""

end