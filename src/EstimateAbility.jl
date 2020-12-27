"""
    _eap(post, gh)
Take the expectation of the posterior distribution for each dimensions.

# Usage
```julia
[_eap(post[p,:], gh[p], bgh) for p in axes(post, 1)]
```
"""
function _eap(post, gh, bgh)
    μ = (gh.T*bgh.n.-gh.m)'post
    deviance = bgh.n' .- μ
    # σ = diagm(sqrt.(deviance .^2 * post))
    Σ = deviance * (deviance' .* post)
    return (μ, Σ)
end

