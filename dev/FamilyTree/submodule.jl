
"""
	get_submodule(m::Module)
	
	Return all the submodules of a given module. children module of m and 
	even children's offsprings as well
"""
function get_submodule(m::Module)
    module_names=names(m)
    submodule=[]
	    for i in module_names
			mod = getfield(m, i)
			if typeof(mod)==Module && mod != m
				push!(submodule, mod)
			end
	    end
	return submodule
end

"""
	get_childrenmodule(m::Module)
	
	Return all the children module of a given module. 
	Only the daughter and son of module m
"""

function get_childrenmodule(m::Module)
	submodule = get_submodule(m)
	grandchildrenmodule = []
	for mod in submodule
		grandchildren = get_submodule(mod)
		if grandchildren != []
			append!(grandchildrenmodule, grandchildren)
		end
	end
	# rempve grandchildren from submodule
	return setdiff(submodule, grandchildrenmodule)
end

"""

	get_struct(m::Module)
	
	Return all the struct names of a given module.
"""

function get_struct(m::Module)
	struct_names = []
	for i in names(m)
		if typeof(getfield(m, i)) == DataType || typeof(getfield(m, i)) == UnionAll
			push!(struct_names, i)
		end
	end
	return struct_names
end

using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end

# Example usage
m = HokseonReproduce
T = HokseonReproduce.Model
module_names=names(m)

i = module_names[2]
getfield(m, i)
getfield(m, i) <: T
M = get_childrenmodule(HokseonReproduce)




