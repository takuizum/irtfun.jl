"""
Compute the expected count of each item categories.
For the dentype = :DN
"""
function expectedcount(post, pind, nind, iind, pars, U)
    N = zeros(Float64, size(pars, 1), size(post, 2))
    r = [zeros(Float64, length(unique(k)), size(post, 2)) for k in eachcol(U)]
    for (i, par) in enumerate(pars)
        group = par.group
        ind = vcat([pind[g] for g in group])[1]
        respind = [i ∈ iind[p] for p in ind]
        ind = ind[respind]
        x::Array{Int64, 1} = @views U[findall(respind), i] .+ one(eltype(U[:, i])) # response vector (original data are starts from 0.)
        for (s::Int64, p::Int64) in zip(x, ind)
            r[i][s, :] += @views post[p, :] .* nind[p]
            N[i, :] += @views post[p, :] .* nind[p]
        end
    end
    return N, r
end

@inline function mvn(x)::eltype(x) 
    length(x) == 1 ? pdf(Normal(), x)[1] : pdf(MvNormal(length(q), 1.0), x)[1] # nomal density
end

"""
Expected log likelihood function. For DN mode.
"""
function ell(newpar, pars, bgh::DNq, r, N, q)::eltype(newpar)
    lnp::eltype(newpar) = zero(eltype(newpar))
    for k in axes(r, 1)
        for l in axes(bgh.n, 1)
            p = @views irf(newpar, pars, bgh.n[l, :], k-1, q)
            if p < 0.0
                @error "Calculated probability is negative ($p)! See your model setting or data."
            end
            lnp += log(p) * r[k, l] / N[l]
        end
    end
    if typeof(pars) <: guessing
        lnp += log(pdf(Beta(2, 8), newpar[end]))
    end
    if typeof(pars) <: LPE
        lnp += log(pdf(Gamma(2.0, 1.0), exp(newpar[end])))
    end
    return lnp
end


"""
Optimize by Newton method.
"""
function Newton(pars, g, H, j, q; atol = 1e-5, N = 5)
    pars′ = take(pars[j], q) - H(j)\g(j)
    t = 0
    while !all(abs.(take(pars[j], q) .- pars′) .< atol) && t ≤ N
        t += 1
        distribute!(pars′, pars[j], q)
        pars′ = take(pars[j], q) - H(j)\g(j)
    end
    distribute!(pars′, pars[j], q)
end