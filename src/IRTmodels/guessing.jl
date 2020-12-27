# Base model struct
Base.@kwdef mutable struct guessing{T <: AbstractFloat} <: IRTmodel
    a::AbstractVector{T} = [0.0]
    d::AbstractVector{T} = [0.0]
    c::AbstractVector{T} = [0.0]
    group::Vector{Int64} = [1]
    fixed::Bool = false
    estc::Bool = true
end

function _distribute!(new, old::guessing)
    old.a = new[begin:length(old.a)]
    old.d = [new[length(old.a)+1]]
    old.c = [new[end]]
end

function _distribute(new, old::guessing)
    out = copy(old)
    out.a = new[begin:length(out.a)]
    out.d = [new[length(out.a)+1]]
    out.c = [new[end]]
    return out
end

function _take(par::guessing)
    if par.estc == true
        return [par.a; par.d; par.c]
    else
        return [par.a; par.d]
    end
end

function _take(par::guessing, q)
    if par.estc == true
        return [par.a[q]; par.d; par.c]
    else
        return [par.a[q]; par.d]
    end
end

function _copy(x::guessing)
    guessing(a = x.a, d = x.d, c = x.c, group = x.group, fixed = x.fixed, estc = x.estc)
end

function _rescale!(par::guessing, μ, Σ, q)
    if length(q) == 1
        # This item follows unidimensional model.
        A = 1/ sqrt(Σ[1]); K = -A*μ[1]
        par.a = par.a / A
        par.d = @. par.d - par.a*K/A
        par.c = [par.c[1]]
    else 
        # This item seems to follow the multidimensional model.
        L = cholesky(Σ).L
        par.a = par.a * L # Dimension mismatch ?
        par.d = @. par.d + par.a * μ
        par.c = [par.c[1]] 
    end
end

function _irf(p::guessing, θ, u, q)::Float64
    if u == 0
        return 1.0 - p.c[1] - (1.0 - p.c[1])*logistic(p.a[q]'θ[q] + p.d[1])
    else
        return p.c[1] + (1.0 - p.c[1])*logistic(p.a[q]'θ[q] + p.d[1])
    end
end

function _irf(p, m::guessing, θ, u, q)::eltype(p)
    a = p[1:length(m.a[q])]
    d = p[length(m.a[q])+1]
    c = m.estc ? p[end] : 0.0
    if u == 0
        return 1.0 - c[1] - (1.0 - c[1])*logistic(a[q]'θ[q] + d[1])
    else
        return c[1] + (1.0 - c[1])*logistic(a[q]'θ[q] + d[1])
    end
end
