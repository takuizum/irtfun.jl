# Base model struct
Base.@kwdef mutable struct graded{T <: AbstractFloat} <: IRTmodel
    a::AbstractVector{T} = [0.0]
    d::AbstractVector{T} = [0.0]
    group::Vector{Int64} = [1]
    fixed::Bool = false
    # graded(;a = [0.0], d = [0.0], group = [1], fixed = true) = new(a, d, group, fixed)
end

function _distribute!(new, old::graded)
    old.a = new[begin:length(old.a)]
    old.d = new[length(old.a)+1:end]
end

function _distribute!(new, old::graded, q)
    old.a = new[begin:length(old.a)][q]
    old.d = new[length(old.a)+1:end]
end

function _distribute(new, old::graded)
    out = copy(old)
    out.a = new[begin:length(out.a)]
    out.d = new[length(out.a)+1:end]
    return out
end

function _distribute(new, old::graded, q)
    out = copy(old)
    out.a = new[begin:length(out.a)][q]
    out.d = new[length(out.a)+1:end]
    return out
end

function _take(par::graded)
    return [par.a; par.d]
end

function _take(par::graded, q)
    return [par.a[q]; par.d]
end

function _copy(x::graded)
    graded(a = x.a, d = x.d, group = x.group, fixed = x.fixed)
end

function _rescale!(par::graded, μ, Σ, q)
    if length(q) == 1
        # This item follows unidimensional model.
        A = 1/ sqrt(Σ[1]); K = -A*μ[1]
        par.a = par.a / A
        par.d = @. par.d - par.a*K/A
    else 
        # This item seems to follow the multidimensional model.
        L = cholesky(Hermitian(Σ)).L
        tmp = par.a' * L
        par.a =  tmp'# Dimension mismatch ?
        par.d = par.d .+ par.a' * μ 
    end
end

function _irf(p::graded, θ, u, q)::Float64
    if u == 0
        return 1.0 - logistic(p.a[q]'θ[q] + p.d[u+1])
    elseif u == length(p.d)
        return logistic(p.a[q]'θ[q] + p.d[u])
    else
        return logistic(p.a[q]'θ[q] + p.d[u]) - logistic(p.a[q]'θ[q] + p.d[u+1])
    end
end

function _irf(p, m::graded, θ, u, q)::eltype(p)
    a = p[1:length(m.a[q])]
    d = p[length(m.a[q])+1:end]
    if u == 0
        return 1.0 - logistic(a'θ[q] + d[u+1])
    elseif u == length(d)
        return logistic(a'θ[q] + d[u])
    else
        return logistic(a'θ[q] + d[u]) - logistic(a'θ[q] + d[u+1])
    end
end
