"""
    BAEstep!(lnL, ind, nod, w, U, pars, Nx, Q)
Compute E step proposed by Bock and Aitkin (1981). The integral is apploximated by discrete normal quadrature.
This function id in-place type.

Non adaptive, non GH version

# Arguments

`lnL` A p(n of subjects) × l(n of nodes) matrix of log likelohood.
`post` A p × l matrix of the apprroximated posterior density.
`pind` Array of arrays that contain the locations of person in each groupes.
`iind` Array of arrays that contains the item location of each records.
`gh` Vector of DNq.
`U` A matrix of item responses.
`pars` Array of IRT model parameters.
`Q` Q matrix.
"""
function BAEstep!(lnL, post, pind, iind, gh::Array{DNq}, U, pars, Q)
    # Reset
    for i in axes(lnL, 1), j in axes(lnL, 2)
            @fastmath @inbounds lnL[i, j] = 0.0
    end
    for g in axes(pind, 1)
        @fastmath @inbounds Threads.@threads for p in pind[g] # person index (Vector of Vector)
            for i in iind[p] # item index (Vector of Vector)
                if U[p,i] !== missing
                    for l::Int64 in axes(gh[g].w, 1)
                        lnL[p,l] += @views log(irf(pars[i], gh[g].n[l,:], U[p,i], Q[i,:])) 
                    end
                end # of l
            end # of item
            # approximate posterior distribution
            P = @fastmath @views exp.(lnL[p,:]) .* gh[g].w
            post[p,:] = @fastmath P ./ sum(P)
        end # of person
    end # of group
    return nothing
end

function initializematrix(P, I; use_static = false)
    if use_static
        return @MMatrix zeros(Float64, P, I)
    else
        return zeros(Float64, P, I)
    end
end
