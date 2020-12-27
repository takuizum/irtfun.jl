
"""
    takeitemindex(u)
Take the index of the item response for each subjects. `u` is a vector of item response.

# Usage
```julia
[takeitemindex(@view U[i,:]) for i in axes(U, 1)]
```
"""
function takeitemindex(u)
    findall(u .!== missing)
end

function takeitemindex(u::DataFrameRow)
    findall([u[i] !== missing for i in axes(u)[1]])
end

"""
    takepersonindex(g::Vector{Int64})
Take the index of the group where person belongs to.

# Usage
```julia
takepersonindex(g)
```
"""
function takepersonindex(g)
    if eltype(g) <: Real
        return [findall(grp .== g) for grp in sort(unique(g))], g
    elseif eltype(g) <: AbstractString
        u = sort(unique(g))
        d = Dict(zip(u, 1:length(u)))
        newind = [d[i] for i in g]
        return [findall(grp .== newind) for grp in sort(unique(newind))], newind
    else
        error("eltype $(eltype(g)) as the group index was not supprted now.")
    end
end

