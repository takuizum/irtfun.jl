# GaussQuadrature.jl
mutable struct DNq
    n::Array{Float64, 2}
    w::Array{Float64, 1}
end

"""
    initializenode(ndims::Int, quadpts::Int)
Initialize the nodes and weights of Gauss-Hermite quadrature (Multidimentional version).
Generate `ndims` dimensional quadrature nodes and weights.

# Arguments

# Optional arguments

- `dentype`
    - `:DN` Discrete normal
    - `:GH` Gauss-Hermite quadrature

# Example
```julia
ghn = initializenode(3, 21)
```
"""
function initializenode(ndims, quadpts; θrange = (-4, 4))
    nodes = range(θrange[1], stop = θrange[2], length = quadpts) 
    weights = pdf.(Normal(), nodes) ./ sum(pdf.(Normal(), nodes))
    ghn = Dict(zip(1:quadpts, nodes))
    ghw = Dict(zip(1:quadpts, weights))
    # https://stackoverflow.com/questions/54256481/converting-array-of-cartesianindex-to-2d-matrix-in-julia
    as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...))
    idx = convert(Matrix, as_ints(vec(CartesianIndices(Array{Float64}(undef, Tuple(fill(quadpts, ndims))))))')
    n = [ghn[idx[i, j]] for i in axes(idx, 1), j in axes(idx, 2)]
    w = prod([ghw[idx[i, j]] for i in axes(idx, 1), j in axes(idx, 2)], dims = 2)[:]
    return DNq(n, w)
end
