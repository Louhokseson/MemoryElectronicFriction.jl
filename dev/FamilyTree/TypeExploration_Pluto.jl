### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 9b00fb5c-f8c6-11ee-0d57-3dbcf47bf546
using NQCDynamics, NQCModels, AbstractTrees, GraphMakie, CairoMakie, LayeredLayouts, Graphs, NetworkLayout

# ╔═╡ a1ea5121-7587-4b31-a0c6-bf1ceaa48d79
begin
	export_theme = Theme(
		palette = (color = [colorant"#c88033", colorant"#4d8b31", colorant"#1d3461", colorant"#00a9ce"],),
		
)
end

# ╔═╡ 5c2886e0-5d52-4d8b-bac7-ba60bbefe31f
begin
	subtypes(NQCModels.Model)
	subtypes(NQCModels.AdiabaticModels.AdiabaticModel)
end

# ╔═╡ 2c783953-7069-49b7-bb0e-b48aae9cc5db
# Define what the Module tree should look like
begin 
	function get_subtypes(m::Module, T::Type)
    module_names=names(m)
    submodules=[]
	    for i in module_names
	        if isa(getfield(m, i), T)
	            if i==Symbol(m) || i==nameof(m)
	            else
	                push!(submodules, getfield(m, i))
	            end
	        end
	    end
	    return submodules
	end
	#AbstractTrees.children(d::DataType) = subtypes(d)
	#AbstractTrees.parent(d::Type) = parentmodule(d)
	AbstractTrees.children(d::Type) = subtypes(d)
	#AbstractTrees.children(m::Module) = m==NQCModels ? get_subtypes(m, Union{Module, DataType}) : get_subtypes(m, DataType)
	AbstractTrees.children(m::Module) = get_subtypes(m, Union{Module})
	#AbstractTrees.parent(m::Module) = parentmodule(m)
end

# ╔═╡ 5b7fca7b-0639-4499-b02d-c70386743e77
begin 
	function build_type_relations_graph(selected_type; include_supertypes=false)
		supertype_list = include_supertypes ? supertypes(selected_type) : nothing
		all_subtypes = AbstractTrees.children(selected_type)
		tree_finished = false
		# Find descendants until done. 
		depth = 0
		while true
			depth+=1
			println("Depth level: $(depth)")
			next_level_subtypes = vcat(all_subtypes, [AbstractTrees.children(current_type) for current_type in all_subtypes]...)
			if symdiff(all_subtypes, next_level_subtypes) != []
				all_subtypes = vcat(all_subtypes, next_level_subtypes)
				unique!(all_subtypes)
			else
				println("Type exploration done")
				break
			end
		end
		# Build a list of all nodes in the type graph and define as a directed graph
		nodes = include_supertypes ? vcat(supertype_list, [selected_type], all_subtypes) : vcat([selected_type], all_subtypes)
		final_graph = SimpleDiGraph(length(nodes))
		# For each type of node, find its children nodes and add their connections
		for (current_index, current_connection) in enumerate(nodes)
			type_children = AbstractTrees.children(current_connection)
			for child in type_children
				connection = findfirst(isequal(child), nodes)
				connection===nothing ? continue : add_edge!(final_graph, current_index => connection)
			end
		end
		return (nodes, final_graph)
	end
end

# ╔═╡ fe5d0800-7958-4a9c-ac77-e2d30f4205c2
all_types, graph = build_type_relations_graph(NQCDynamics.NQCModels.Model; include_supertypes=false)

# ╔═╡ f9e7a9b0-12ed-4d59-9fd9-819e9dba3207
begin 
	graph_layout = solve_positions(Zarate(), graph)
	layout = Point2f.(zip(graph_layout[1], graph_layout[2]))
	waypoints = [Point2f.(zip(graph_layout[3][e]...)) for e in edges(graph)]
end

# ╔═╡ ce752108-2346-4faa-89ea-8e94a90829e3
begin
	rectangle(width, height) = Makie.Polygon([Point2f(-width/2,-height/2), Point2f(width/2,-height/2), Point2f(width/2,height/2), Point2f(-width/2,height/2)])
	figure, ax, graphplotseries=graphplot(
		graph;
		figure = (fonts=(; regular="Lexend"), ),
		layout = layout,
		#layout = Buchheim(),
		waypoints = waypoints,
		#layout = Stress(; weights=[length(string.(Symbol.(i))*string.(Symbol.(j))) for i in all_types, j in all_types]),
		ilabels = last.(split.(string.(Symbol.(all_types)), ".")),
		ilabels_fontsize = 8,
		ilabels_color="white",
		#ilabels_attr = (; overdraw=true),
		#nlabels_offset = Point2f(-0.25 ,0.1),
		edge_color="grey",
		node_attr = (color=colorant"#0b6e4f", marker=rectangle(2.2, 0.28), markerspace=:pixel, markersize=50),
	)
	hidespines!(ax)
	hidedecorations!(ax)
	xlims!(0.65,3.5)
	save("Test.svg", figure)
	figure
end

# ╔═╡ 0fe9d764-6a0d-4ea8-8c5c-d6ecea377c0f
string(Symbol(NQCModels.Model))

# ╔═╡ 358857da-231b-4df0-a48b-677ae085df14
begin
	# Explore down a module
	module_to_analyse=NQCModels
	
	module_names=names(module_to_analyse)
	submodules = []
	functions = []
	
	for i in module_names
		sub=getfield(module_to_analyse, i)
		if sub==module_to_analyse
			print("Module self-reference")
		elseif isa(sub, Module)
			push!(submodules, sub)
		elseif isa(sub, Function)
			push!(functions, sub)
		end
	end
	println(submodules)
	println(functions)
end

# ╔═╡ e1fe16e0-cb37-46e9-8d52-4442a5334856
print_tree(NQCDynamics.Calculators.AbstractCalculator)

# ╔═╡ 59dd45a4-9227-4a2f-a549-4fb2955865c3
names(NQCModels)

# ╔═╡ 6969e41d-3f0e-4696-9253-35388b92c084
AbstractTrees.children(Any)

# ╔═╡ 18479d49-8dc7-4513-8eda-7c54228e5415
get_subtypes(NQCModels, Module)

# ╔═╡ b11b87ba-94ed-4a92-ae66-7d3a3a284869
names(NQCModels)[25] == Symbol(NQCModels)

# ╔═╡ 16e95fb7-1bc0-49ab-892a-1c09958e56cd
names(NQCModels.AdiabaticModels)[2] == nameof(NQCModels.AdiabaticModels)

# ╔═╡ b4d2319d-8ed7-443e-9ad6-a89beb5bd812


# ╔═╡ 74d92dcc-3ebb-4934-9fc4-eb683f9af1ec
findall(isequal(:NQCModels), names(NQCModels))

# ╔═╡ c3a76434-a607-49af-a5dd-c542825e94f0
supertype(AdiabaticModels.AdiabaticModel)

# ╔═╡ a566d261-025d-4476-8a99-8d92a0165cfc
begin
l1c = AbstractTrees.children(NQCModels.Model)
l2c = vcat([AbstractTrees.children(i) for i in unique(vcat(NQCModels.Model, l1c))]...)
l3c = vcat([AbstractTrees.children(i) for i in unique(vcat(NQCModels.Model, l1c, l2c))]...)
l4c = vcat([AbstractTrees.children(i) for i in unique(vcat(NQCModels.Model, l1c, l2c, l3c))]...)
l5c = vcat([AbstractTrees.children(i) for i in unique(vcat(NQCModels.Model, l1c, l2c, l3c, l4c))]...)
	symdiff(l5c, l4c) == []
end

# ╔═╡ Cell order:
# ╠═9b00fb5c-f8c6-11ee-0d57-3dbcf47bf546
# ╠═a1ea5121-7587-4b31-a0c6-bf1ceaa48d79
# ╠═5c2886e0-5d52-4d8b-bac7-ba60bbefe31f
# ╠═2c783953-7069-49b7-bb0e-b48aae9cc5db
# ╠═5b7fca7b-0639-4499-b02d-c70386743e77
# ╠═fe5d0800-7958-4a9c-ac77-e2d30f4205c2
# ╠═f9e7a9b0-12ed-4d59-9fd9-819e9dba3207
# ╠═ce752108-2346-4faa-89ea-8e94a90829e3
# ╠═0fe9d764-6a0d-4ea8-8c5c-d6ecea377c0f
# ╠═358857da-231b-4df0-a48b-677ae085df14
# ╠═e1fe16e0-cb37-46e9-8d52-4442a5334856
# ╠═59dd45a4-9227-4a2f-a549-4fb2955865c3
# ╠═6969e41d-3f0e-4696-9253-35388b92c084
# ╠═18479d49-8dc7-4513-8eda-7c54228e5415
# ╠═b11b87ba-94ed-4a92-ae66-7d3a3a284869
# ╠═16e95fb7-1bc0-49ab-892a-1c09958e56cd
# ╠═b4d2319d-8ed7-443e-9ad6-a89beb5bd812
# ╠═74d92dcc-3ebb-4934-9fc4-eb683f9af1ec
# ╠═c3a76434-a607-49af-a5dd-c542825e94f0
# ╠═a566d261-025d-4476-8a99-8d92a0165cfc
