"""
Initialize model parameters (in-place).
"""
init!(x::IRTmodel, q, u) = _init!(x, q, u)
init(x::IRTmodel, q, u) = _init(x, q, u)

# graded
function _init!(x::graded, q, u)
    # u is a vector of item response
    freq = map(j -> count(i -> i == j, u), sort(unique(skipmissing(u))))
    prop = freq ./ sum(freq)
    b = @. quantile(Normal(), cumsum(prop))
    d = -b[begin:1:end-1]
    a = fill(1.5, length(q))
    a[.!q] .= 0.0
    x.a = a
    x.d = d
    x.group = [1]
    x.fixed = false
end

function _init(x::graded, q, u)
    # u is a vector of item response
    freq = map(j -> count(i -> i == j, u), sort(unique(skipmissing(u))))
    prop = freq ./ sum(freq)
    b = @. quantile(Normal(), cumsum(prop))
    d = -b[begin:1:end-1]
    a = fill(1.5, length(q))
    a[.!q] .= 0.0
    return graded(a = a, d = d, group = [1], fixed = false)
end

"""
    setgroupindex!(x::IRTmodel, u, g)
Set the group index to the struct::IRTmodel.
"""
function setgroupindex!(x::IRTmodel, u, g)
    group = unique(g[u .!== missing])
    x.group = group
end

# guessing

function _init!(x::guessing, q, u)
    prop = mean(u)
    d = [-quantile(Normal(), prop)]
    a = fill(1.5, count(q))
    x.a = a
    x.d = d
    x.c = [c]
    x.group = [1]
    x.fixed = false
    x.estc = true
end

function init(x::guessing, q, u)
    prop = mean(u)
    d = [-quantile(Normal(), prop)]
    a = fill(1.5, count(q))
    return guessing(a = a, d = d, c = [0.2], group = [1], fixed = false, estc = true)
end


# LPE
function _init!(x::LPE, q, u)
    # u is a vector of item response
    freq = map(j -> count(i -> i == j, u), sort(unique(skipmissing(u))))
    prop = freq ./ sum(freq)
    b = @. quantile(Normal(), cumsum(prop))
    d = -b[begin:1:end-1]
    a = fill(1.5, length(q))
    a[.!q] .= 0.0
    x.a = a
    x.d = d
    x.ξ = [log(1.0)]
    x.group = [1]
    x.fixed = false
end

function _init(x::LPE, q, u)
    # u is a vector of item response
    freq = map(j -> count(i -> i == j, u), sort(unique(skipmissing(u))))
    prop = freq ./ sum(freq)
    b = @. quantile(Normal(), cumsum(prop))
    d = -b[begin:1:end-1]
    a = fill(1.5, length(q))
    a[.!q] .= 0.0
    return LPE(a = a, d = d, ξ = [log(1.0)], group = [1], fixed = false)
end