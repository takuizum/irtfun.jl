# Models.jl
"""
Type of IRT models. User-defined model can be used.
A struct which hyper type is `IRTmodel` must contain these containers below:

- Model parameter(s)
- group
- fixed

Then, some methods should be implemented:

- `_take` Extract model item parameter to be estimated in EM cycle.
- `_distribute` and `_distribute!` Attach new value to contaiers of parameters.
- `_irf` Item response function. 2 type of return value, one is Float64, 
anothoer is eltype(par), which is used in ForwardDiff, should be implemented.
- `_init` and `_init!` Initialize item parameters, which are used for the starting value in the estimation.
- `_rescale!` Rescaling item parameters at each EM cycles.
- `_checkitem` Check whether estimated parameters are taking appropriate values.

See "IRTmodels/graded.jl" for an example.

"""
abstract type IRTmodel end

"""
Take model paraemters.
"""
function take(par::IRTmodel)
    _take(par)
end

"""
Take model parameters based on `q` matrix(vector).
"""
function take(par::IRTmodel, q)
    _take(par, q)
end

"""
Distribute `new` value to `old` as model parameters.
"""
function distribute!(new, old::IRTmodel)
    _distribute!(new, old)
end

function distribute!(new, old::IRTmodel, q)
    _distribute!(new, old, q)
end

function distribute(new, old::IRTmodel)::IRTmodel
    _distribute(new, old)
end

function distribute(new, old::IRTmodel, q)::IRTmodel
    _distribute(new, old, q)
end

function copy(x::IRTmodel)
    _copy(x)
end

function rescale!(pars::IRTmodel, μ, Σ, q)
    _rescale!(pars, μ, Σ, q)
end

"""
    irf(p::graded, θ, u, q)
Item respose function

# Arguments

`p` Probability model contains its parameters.
`θ` A latent trait vector
`q` Q matrix(vector).
`u` An item response.

# Example
```julia
# Graded response model
pars = graded(a = 1.0, d = [2, 0, -1.5])
[irf.(Ref(pars), -4:0.1:4, k, 1) for k in 0:3] |> plot
```
"""
function irf(p::IRTmodel, θ, u, q)
    _irf(p, θ, u, q)
end

# For the internal use in ForwardDiff 
function irf(p, m::IRTmodel, θ, u, q)
    _irf(p, m, θ, u, q)
end
