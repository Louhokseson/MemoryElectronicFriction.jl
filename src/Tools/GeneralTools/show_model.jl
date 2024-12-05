function Base.show(io::IO, m::T) where T
    if T <: BrandbygeAborbate
        println(io, "$T(")
        for field in fieldnames(T)
            println(io, "    $field = $(getfield(m, field)),")
        end
        print(io, ")")
    else
        Base.show_default(io, m)
    end
end
