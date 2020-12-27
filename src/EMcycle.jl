
"""
Run one EM cycle.

# Arguments

- `lnL` A matrix of the log likelihood at the discrete points.
- `post` A matrix of the mass of the posterior distribusion at the discrete points.
- `pind` Array of array which contains the locations of all records. The array of locations is stored by each groups where records bnelong to.
- `nind` Matrix contains the observed counts of all records.
- `iind` Array of array which the item location responded to for all records.
- `gh` Array of Gauss-Hermite quadrature's nodes and weights. One set of nodes and weights are prepared for each records.
- `U` Matrix of item response.
- `pars` Provisional item parameters.
- `Q` Q matrix.
"""
function BAEMcycle(lnL, post, pind, nind, iind, mc, bgh::GHq, U, pars, Q; Mstep = Mstep())
    BAEstep!(lnL, post, pind, iind, mc, bgh::GHq, U, pars, Q)
    g = (j) -> gradient(x′ -> ell(x′, pars[j], mc, bgh, post, pind, nind, Q[j,:], @view U[:,j]), take(pars[j], Q[j,:]))
    H = (j) -> hessian(x′ -> ell(x′, pars[j], mc, bgh, post, pind, nind, Q[j,:], @view U[:,j]), take(pars[j], Q[j,:]))
    Threads.@threads for i in axes(pars, 1)
        if !pars[i].fixed
            @fastmath @inbounds Newton(pars, g, H, i, Q[i,:]; atol = Mstep.atol, N = Mstep.N)
        end
    end
    return nothing, nothing
end

function BAEMcycle(lnL, post, pind, nind, iind, gh, bgh::DNq, U, pars, Q; Mstep = Mstep())
    BAEstep!(lnL, post, pind, iind, gh, U, pars, Q)
    N, r = expectedcount(post, pind, nind, iind, pars, U)
    # g = (j) -> gradient(x′ -> ell(x′, pars[j], bgh, r[j], Q[j,:]), take(pars[j], Q[j,:]))        
    # H = (j) -> hessian(x′ -> ell(x′, pars[j], bgh, r[j], Q[j,:]), take(pars[j], Q[j,:]))
    Threads.@threads for i in axes(pars, 1)
        if !pars[i].fixed
            if Mstep.method == :Newton
                @fastmath @inbounds Newton(pars, g, H, i, Q[i,:]; atol = Mstep.atol, N = Mstep.N)
            else
                res = optimize(x′ -> -ell(x′, pars[i], bgh, r[i], Q[i,:]), take(pars[i], Q[i,:]), Mstep.method)
                distribute!(minimizer(res), pars[i], Q[i,:])
            end
        end
    end
    return N, r
end

"""
Check whether item parameters seems to converged
"""
function checkconvergence(new, old; atol = 1e-3)
    diff = [maximum(abs.(take(new[i]) .- take(old[i]))) for i in axes(new, 1)]
    print("Largest diff ", @sprintf "%1.5f" maximum(diff))
    return diff .< atol
end