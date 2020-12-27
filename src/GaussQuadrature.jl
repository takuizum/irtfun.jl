# GaussQuadrature.jl

mutable struct GHq
    n::Array{Float64, 2}
    w::Array{Float64, 1}
end

mutable struct DNq
    n::Array{Float64, 2}
    w::Array{Float64, 1}
end

mutable struct mcov
    m::Array{Float64, 1}
    v::Array{Float64, 2}
    T::Cholesky{Float64,Array{Float64,2}}
    mcov(q; m = fill(0.0, q), v = diagm(fill(1.0, q)), T = cholesky(v)) = new(m ,v, T)
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
function initializenode(ndims, quadpts; dentype = :DN, θrange = (-4, 4))
    if !(dentype ∈ [:DN, :GH])
        error("Inappropriate dentype $(dentype) were specified.")
    end
    if dentype == :DN
        nodes = range(θrange[1], stop = θrange[2], length = quadpts) 
        weights = pdf.(Normal(), nodes) ./ sum(pdf.(Normal(), nodes))
        ghn = Dict(zip(1:quadpts, nodes))
        ghw = Dict(zip(1:quadpts, weights))
    elseif dentype == :GH
        nodes, weights = gausshermite(quadpts)
        # Scale is mean = 0 and sd = 1
        # ghn = Dict(zip(1:quadpts, nodes./√2))
        # ghw = Dict(zip(1:quadpts, weights.*exp.(nodes.^2)))
        ghn = Dict(zip(1:quadpts, nodes))
        ghw = Dict(zip(1:quadpts, weights))
    end
    # https://stackoverflow.com/questions/54256481/converting-array-of-cartesianindex-to-2d-matrix-in-julia
    as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...))
    idx = convert(Matrix, as_ints(vec(CartesianIndices(Array{Float64}(undef, Tuple(fill(quadpts, ndims))))))')
    n = [ghn[idx[i, j]] for i in axes(idx, 1), j in axes(idx, 2)]
    w = prod([ghw[idx[i, j]] for i in axes(idx, 1), j in axes(idx, 2)], dims = 2)[:]
    if dentype == :DN
        return DNq(n, w)
    elseif dentype == :GH
        return GHq(n, w)
    end
end

"""
Change nodes and weights to be adaptive.

The moments of the latent distribution are estiamted by mean and variance of the posterior distribution.

(Note) Mode and the inverse of informaton are more efficient ? 
"""
function adapt!(mc::mcov, moments)
    mc.m = moments[1]
    mc.v = sqrt.(moments[2])
    mc.T = cholesky(mc.v)
end