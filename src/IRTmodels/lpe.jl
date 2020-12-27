# Logistic exponential model

# Base model struct
Base.@kwdef mutable struct LPE{T <: AbstractFloat} <: IRTmodel
    a::AbstractVector{T} = [0.0]
    d::AbstractVector{T} = [0.0]
    ξ::AbstractVector{T} = [0.0]
    group::Vector{Int64} = [1]
    fixed::Bool = false
end

function _distribute!(new, old::LPE)
    old.a = new[begin:length(old.a)]
    old.d = [new[length(old.a)+1]]
    old.ξ = [new[end]]
end

function _distribute(new, old::LPE)
    out = copy(old)
    out.a = new[begin:length(out.a)]
    out.d = [new[length(out.a)+1]]
    out.ξ = [new[end]]
    return out
end

function _distribute!(new, old::LPE, q)
    old.a = new[begin:length(old.a[q])]
    old.d = [new[length(old.a[q])+1]]
    old.ξ = [new[end]]
end

function _distribute(new, old::LPE, q)
    out = copy(old)
    out.a = new[begin:length(out.a[q])]
    out.d = [new[length(out.a[q])+1]]
    out.ξ = [new[end]]
    return out
end

function _take(par::LPE)
    return [par.a; par.d; par.ξ]
end

function _take(par::LPE, q)
    return [par.a[q]; par.d; par.ξ]
end

function _copy(x::LPE)
    LPE(a = x.a, d = x.d, ξ = x.ξ, group = x.group, fixed = x.fixed)
end

function _rescale!(par::LPE, μ, Σ, q)
    if length(q) == 1
        # This item follows unidimensional model.
        A = 1/ sqrt(Σ[1]); K = -A*μ[1]
        par.a = par.a / A
        par.d = @. par.d - par.a*K/A
        par.ξ = [par.ξ[1]]
    else 
        # This item seems to follow the multidimensional model.
        L = cholesky(Σ).L
        par.a = par.a * L # Dimension mismatch ?
        par.d = @. par.d + par.a * μ
        par.ξ = [par.ξ[1]] 
    end
end

function _irf(p::LPE, θ, u, q)::Float64
    if u == 0
        return 1.0 - (logistic(p.a[q]'θ[q] + p.d[1]))^exp(p.ξ[1])
    else
        return (logistic(p.a[q]'θ[q] + p.d[1]))^exp(p.ξ[1])
    end
end

function _irf(p, m::LPE, θ, u, q)::eltype(p)
    a = p[1:length(m.a[q])]
    d = p[length(m.a[q])+1]
    ξ = exp(p[end])
    if u == 0
        return 1.0 - (logistic(a[q]'θ[q] + d[1]))^ξ
    else
        return (logistic(a[q]'θ[q] + d[1]))^ξ
    end
end
