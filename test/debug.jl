using irtfun
using RCall, DataFrames

# Single group
Science = R"""
library(mirt)
Science
"""
Science = convert(Matrix, Science)
R"""
coef(mirt(Science, 1))
"""
@profview estimate(Science, 1, graded(), E = Estep(quadpts = 21))
@time estimate(Science, 1, graded(), E = Estep(quadpts = 61))
est_gh = estimate(Science, 1, graded(), E = Estep(quadpts = 5, dentype = :GH))
# mirt
est_nd = estimate(Science, 1, graded(), E = Estep(quadpts = 61))
est_nd = estimate(Science, 2, graded(), E = Estep(quadpts = 5, dentype = :GH))

# Multple group
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
mod_fullconstrain <- multipleGroup(dat, models, group = group,
invariance=c('slopes', 'intercepts', 'free_mean', 'free_var'))
coef(mod_fullconstrain)
"""
data = DataFrame(convert(Matrix, mgdata[1]))
group = convert(Vector, mgdata[2])

@time estimate(data, 1, graded(); E = Estep(quadpts = 21), M = Mstep(N = 5))
@time estimate(data, 1, graded(); group = group, E = Estep(quadpts = 61, dentype = :DN))
@time estimate(data, 1, LPE(); group = group, E = Estep(quadpts = 61, dentype = :DN))
@time estimate(data, 1, graded(); group = group, E = Estep(quadpts = 21, dentype = :GH))

# Guessing model
using RCall, DataFrames
gdata = R"""
library(mirt)
dat <- expand.table(LSAT7)
"""
R"""
coef(mirt(dat, 1, '3PL'))
"""
data = convert(DataFrame, gdata)
@time estimate(data, 1, guessing(); E = Estep(quadpts = 61), M = Mstep(N = 5))



