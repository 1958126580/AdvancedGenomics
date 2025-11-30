"""
    Copula.jl

Copula methods for genomic analysis.
Models dependencies between variables independent of their marginal distributions.
"""

using Statistics
using LinearAlgebra
using Distributions

"""
    pobs(data)

Converts data to pseudo-observations (uniform ranks).
u_ij = rank(x_ij) / (n + 1)
"""
function pobs(data::AbstractMatrix)
    n, m = size(data)
    u = zeros(Float64, n, m)
    for j in 1:m
        u[:, j] = tiedrank(data[:, j]) ./ (n + 1)
    end
    return u
end

# Helper for tiedrank if not available (StatsBase usually has it, but we are using Statistics)
function tiedrank(v::AbstractVector)
    n = length(v)
    p = sortperm(v)
    r = zeros(Float64, n)
    r[p] = 1:n
    # Handle ties? Simple rank for now.
    return r
end

"""
    gaussian_copula_fit(data)

Fits a Gaussian Copula to the data.
Returns the correlation matrix of the Gaussian Copula.
"""
function gaussian_copula_fit(data::AbstractMatrix)
    # 1. Transform to pseudo-observations (Uniform)
    u = pobs(data)
    
    # 2. Transform to Normal quantiles (Inverse CDF)
    # z_ij = Phi^-1(u_ij)
    z = quantile.(Normal(), u)
    
    # 3. Compute Correlation Matrix
    # This captures the dependency structure
    rho = cor(z)
    
    return rho
end
