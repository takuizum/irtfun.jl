module irtfun

using StatsFuns
using FastGaussQuadrature
import Distributions: Normal, MvNormal, Beta, Gamma, pdf, cdf, quantile
using DataFrames: DataFrameRow, DataFrame, All, groupby, nrow, combine, Not
using ForwardDiff: gradient, hessian, Dual
using Printf: @sprintf
using DataFrames: AbstractDataFrame, mapcols
using StaticArrays
using LinearAlgebra: LowerTriangular, diagm, Cholesky, cholesky, det, Hermitian, tril!
using Statistics: mean
using Optim: optimize, BFGS, Newton, minimizer

include("Models.jl")
include("IRTmodels/graded.jl")
include("IRTmodels/guessing.jl")
include("IRTmodels/lpe.jl")
include("GaussQuadrature.jl")
include("Estep.jl")
include("Mstep.jl")
include("EMcycle.jl")
include("StartingValues.jl")
include("index.jl")
include("Rescore.jl")
include("Qmatrix.jl")
include("Estimate.jl")
include("UpdateLatentDistributions.jl")
include("CheckParameters.jl")

export IRTmodel
export irf
export distribute
export distribute!
export take
export rescale!
export graded
export guessing
export LPE

export DNq
export initializenode

export BAEstep!
export initializematrix

export ell
export Newton
export expectedcount

export BAEMcycle
export checkconvergence

export init!
export init

export takeitemindex
export takepersonindex
export setgroupindex!

export rescore

export explanatoryMIRT

export Estep
export Mstep
export estimate

export Moments
export updateAGH!
export updateDN!

export checitemparameters

end
