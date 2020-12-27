
"""
    rescore(x::Vector{Real})
Rescore the vector of item response.

# Usage
```julia
mapcols(c -> rescore(c), U)
```
"""
function rescore(u)
    obs = sort(unique(skipmissing(u)))
    rescored = 0:1:length(obs)-1
    d = Dict(zip(obs, rescored))
    return [o === missing ? missing : d[o] for o in u]
end