
# Test data
using irtfun
using RCall, DataFrames

Science = R"""
library(mirt)
Science
"""
Science = convert(Matrix, Science)

# Graded Response model
using Plots
pars = graded(a = [1.0], d = [2, 0, -1.5])
[irf.(Ref(pars), -4:0.1:4, k, 1) for k in 0:3] |> plot

# 3PLM (guessing) model
pars = guessing(a = [1.0], d = [0.0], c = [0.2])
[irf.(Ref(pars), -4:0.1:4, k, 1) for k in 0:1] |> plot



# Gauss Hermite
using Distributions, FastGaussQuadrature, LinearAlgebra
n, w = gausshermite(20)
# Rescaling
dot(pdf.(Normal(0, 1), n./√2), w.*exp.(n.^2)) / √2

# Cholesky decomposition
using LinearAlgebra
A = [1.0 0.6; 0.6 1.0]
B = cholesky(A)
# B.L'B.L
B.U'B.U


D = combine(groupby([Science DataFrame(__group = fill(1, nrow(Science)))], All()), nrow => :__count)
U = D[:, Not([:__count, :__group])]
F = D[:, :__count]
G = D[:, :__group]
U = mapcols(c -> rescore(c), U) # rescoreは，オプションで選択可能にする。エラー検知もできるとよい。
pind, G = takepersonindex(G)
nind = Dict(zip(axes(U, 1), F))
iind = [takeitemindex(@view U[i,:]) for i in axes(U, 1)]
giind = [findall(map(c -> !all(ismissing(c)), eachcol(U[pind[g], :]))) for g in axes(pind, 1)]
P, I = size(U)
nnode = 5
lnL = initializematrix(P, nnode)
post = initializematrix(P, nnode)
Q = explanatoryMIRT(1, 4)
pars = [init(graded(), Q[i,:], Science[:, i]) for i in 1:I]
[setgroupindex!(pars[j], U[:,j], G) for j in 1:I]
gh = [mcov(size(Q, 2)) for _ in 1:P]
gh = [initializenode(size(Q, 2), nnode) for _ in 1:length(unique(G))]
bgh = initializenode(size(Q, 2), nnode; dentype = :GH)
bgh = initializenode(size(Q, 2), nnode; dentype = :DN)
pmoments = [Moments(size(Q, 2)) for _ in unique(G)]

# EM cycle
BAEMcycle(lnL, post, pind, nind, iind, gh, bgh, U, pars, Q; Mstep = Mstep())
@profview BAEMcycle(lnL, post, pind, nind, iind, gh, bgh, U, pars, Q; Mstep = Mstep())

# Estep
# DN
BAEstep!(lnL, post, pind, iind, gh, U, pars, Q)
@code_warntype BAEstep!(lnL, post, pind, iind, gh, U, pars, Q)
# GH
BAEstep!(lnL, post, pind, iind, gh, bgh, U, pars, Q)
@code_warntype BAEstep!(lnL, post, pind, iind, gh, bgh, U, pars, Q)

# using Plots; plot(bgh.n, post')

N, r = expectedcount(post, pind, nind, iind, pars, U)
@code_warntype expectedcount(post, pind, nind, iind, pars, U)

# Adapt the quadrature points
moments = [_eap(post[p,:], bgh) for p in axes(post, 1)]
[adapt!(gh[p], moments[p]) for p in axes(gh, 1)]

# Estimate the moments of the population
@code_warntype updateDN!(pind, nind, post, gh, pmoments)
updateAGH!(pind, nind, moments, pmoments, 1)

# Mstep

# ForwardDiff version
using ForwardDiff
# DN
@time test = ell([1.2, 0, -1.0, -2.0], pars[1], bgh, r[1], Q[1,:]) # -787くらい
@code_warntype ell([1.2, 0, -1.0, -2.0], pars[1], bgh, r[1], Q[1,:])
g = (j::Int64) -> ForwardDiff.gradient(x′ -> ell(x′, pars[j], bgh, r[1], Q[1,:]), take(pars[j], Q[j,:]))
H = (j::Int64) -> ForwardDiff.hessian(x′ -> ell(x′, pars[j], bgh, r[1], Q[1,:]), take(pars[j], Q[j,:]))
@code_warntype g(1)
@code_warntype Newton(pars, g, H, 1)
@code_warntype H(1)
# GH
@time test = ell([1.2, 0, -1.0, -2.0], pars[1], gh, bgh, post, pind, nind, Q[1,:], U[:, 1]) # -787くらい
@code_warntype ell([1.2, 0, -1.0, -2.0], pars[1], gh, bgh, post, pind, nind, Q[1,:], U[:, 1])
g = (j::Int64) -> ForwardDiff.gradient(x′ -> ell(x′, pars[j], gh, bgh, post, pind, nind, Q[j,:], @view U[:,j]), take(pars[j], Q[j,:]))
H = (j::Int64) -> ForwardDiff.hessian(x′ -> ell(x′, pars[j], gh, bgh, post, pind, nind, Q[j,:], @view U[:,j]), take(pars[j], Q[j,:]))

for j in 1:4
    @time @fastmath Newton(pars, g, H, j, Q[j,:])
end

oldpars = irtfun.copy.(pars)
@code_warntype BAEMcycle(lnL, post, pind, nind, iind, gh, bgh, U, pars, Q)
checkconvergence(pars, oldpars)

# Adaptive quadrature
moments = [_eap(post[p,:], gh[p]) for p in axes(post, 1)]
[adapt!(gh[p], bgh, moments[p]) for p in axes(gh, 1)]
update!(pind, nind, moments, pmoments, size(Q, 2))
[rescale!(pars[i], pmoments[1].μ, pmoments[1].Σ, Q[i,:]) for i in axes(pars, 1)]

@time estimate(Science, 1, graded(), E = Estep(quadpts = 11, N = 10))
@time estimate(Science, 1, graded(), E = Estep(quadpts = 61))
@profview estimate(Science, 1, graded(), E = Estep(N = 10, quadpts = 121))
@code_warntype estimate(Science, 1, graded(), E = Estep(N = 2))

@rlibrary mirt
@time mirt(Science, 1)

# Multiple group
# Test data
using irtfun
using RCall, DataFrames

mgdata = R"""
library(mirt)
set.seed(12345)
a <- matrix(abs(rnorm(15,1,.3)), ncol=1)
d <- matrix(rnorm(15,0,.7),ncol=1)
itemtype <- rep('2PL', nrow(a))
N <- 1000
dataset1 <- simdata(a, d, N, itemtype)
dataset2 <- simdata(a, d, N, itemtype, mu = .1, sigma = matrix(1.5))
dat <- rbind(dataset1, dataset2)
group <- c(rep('D1', N), rep('D2', N))
list(dat, group)
"""
R"""
models <- 'F1 = 1-15'
mod_configural <- multipleGroup(dat, models, group = group)
"""
data = DataFrame(convert(Matrix, mgdata[1]))
group = convert(Vector, mgdata[2])

D = combine(groupby([data DataFrame(__group = group)], All()), nrow => :__count)
U = D[:, Not([:__count, :__group])]
F = D[:, :__count]
G = D[:, :__group]
U = mapcols(c -> rescore(c), U) # rescoreは，オプションで選択可能にする。エラー検知もできるとよい。
pind, G = takepersonindex(G)
nind = Dict(zip(axes(U, 1), F))
iind = [takeitemindex(@view U[i,:]) for i in axes(U, 1)]
P, I = size(U)
lnL = initializematrix(P, 61; use_static = true)
[initializematrix(1, 61; use_static = true) for _ in 1:2000]
post = initializematrix(P, 61; use_static = true)
Q = fill(true, I)
pars = [init(graded(), Q[i,:], data[:, i]) for i in 1:I]
[setgroupindex!(pars[j], U[:,j], G) for j in 1:I]
# gh = [initializenode(size(Q, 2), 21) for _ in 1:P]
gh = [initializenode(size(Q, 2), 61; use_static = true) for _ in 1:length(unique(G))]
bgh = initializenode(size(Q, 2), 61; use_static = true)
pmoments = [Moments(size(Q, 2)) for _ in unique(G)]