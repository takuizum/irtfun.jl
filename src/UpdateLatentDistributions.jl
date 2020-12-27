
mutable struct Moments
    μ
    Σ
    Moments(ndims) = new(zeros(Float64, ndims), diagm(fill(1.0, ndims)))
end

function updateDN!(pind, nind, post, gh, pmoments)
    G = size(pind, 1)
    for g in axes(pind, 1)
        N = 0
        d = @views @fastmath @inbounds sum([post[i, :] .* nind[i] for i in pind[g]])
        d = d ./ sum(d)
        m = gh[g].n'd
        deviance = gh[g].n .- m'
        s = d' * deviance.^2
        pmoments[g].μ = m
        pmoments[g].Σ = s
        if g > 1
            gh[g].w = map(x -> pdf(MvNormal(m, sqrt.(s)), x), eachrow(gh[g].n))
            gh[g].w ./= sum(gh[g].w)
        end
    end
end
