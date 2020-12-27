struct Estep
    N
    method
    quadpts
    atol
    dentype
    Estep(;N = 500, method = :BAEM, quadpts = 21, atol = 1e-5, dentype = :DN) = new(N, method, quadpts, atol, dentype)
end

struct Mstep
    N
    method
    atol
    Mstep(;N = 10, method = BFGS(), atol = 1e-5) = new(N, method, atol)
end

"""
Run EM cycle to estimate the model item parameters from data and Q matrix.
"""
function estimate(D::AbstractDataFrame, ndims::Int64, model::IRTmodel; group = fill(1, size(D, 1)), E::Estep = Estep(), M::Mstep = Mstep())
    if typeof(model) <: IRTmodel
        model = fill(model, size(D, 2))
    end
    Q = explanatoryMIRT(ndims, size(D, 2))
    U = combine(groupby([D DataFrame(__group = group)], All()), nrow => :__count)
    _estimate(U, Q, model, E::Estep, M::Mstep)
end

# Internal function to run EM cycle for the general models.
# This function assume to be passed the multidimensional Q matrix.
function _estimate(D, Q, model, E::Estep, M::Mstep)

    # Initialization step
    U = D[:, Not([:__count, :__group])]
    F = D[:, :__count]
    G = D[:, :__group]
    U = mapcols(c -> rescore(c), U) # rescoreは，オプションで選択可能にする。エラー検知もできるとよい。
    pind, G = takepersonindex(G)
    nind = Dict(zip(axes(U, 1), F))
    iind = [takeitemindex(@view U[i,:]) for i in axes(U, 1)]
    P, I = size(U)
    lnL = initializematrix(P, E.quadpts^size(Q, 2))
    post = initializematrix(P, E.quadpts^size(Q, 2))
    pars = [init(model[i], Q[i,:], D[:, Not([:__count, :__group])][:, i]) for i in 1:I]
    [setgroupindex!(pars[j], U[:,j], G) for j in 1:I]
    ghsize = E.dentype == :DN ? length(unique(G)) : P
    if E.dentype == :GH
        gh = [mcov(size(Q, 2)) for i in axes(U, 1)]
    elseif E.dentype == :DN
        gh = [initializenode(size(Q, 2), E.quadpts; dentype = E.dentype) for i in 1:ghsize]
    end
    bgh = initializenode(size(Q, 2), E.quadpts; dentype = E.dentype) # for reference
    pmoments = [Moments(size(Q, 2)) for _ in unique(G)]
    # Start EM Cycle
    if E.method == :BAEM
        for iter in 1:E.N
            print("EMcycle = ", @sprintf "%4.0f : " iter)
            old = copy.(pars)
            N, r = BAEMcycle(lnL, post, pind, nind, iind, gh, bgh, U, pars, Q; Mstep = M)
            if any(checitemparameters.(pars))
                error("\nStop the estimation.\n")
            end
            if E.dentype == :GH
                moments = [_eap(post[p,:], gh[p], bgh) for p in axes(post, 1)]
                [adapt!(gh[p], moments[p]) for p in axes(gh, 1)]
                updateAGH!(pind, nind, moments, pmoments, size(Q, 2))
            elseif E.dentype == :DN
                updateDN!(pind, nind, post, gh, pmoments)
            end
            @fastmath @inbounds [rescale!(pars[i], pmoments[1].μ, pmoments[1].Σ, Q[i,:]) for i in axes(pars, 1)]
            flag = checkconvergence(old, pars; atol = E.atol)
            print(@sprintf " MOMENTS μ = %1.3f, σ² = %1.3f" pmoments[1].μ[1] pmoments[1].Σ[1])
            if all(flag)
                print("\nConverged.\n\n")
                break
            end
            print("\r")
        end
    end

    # Report estimated parameters, SE, and the other information.
    return pars, pmoments
end
