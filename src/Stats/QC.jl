"""
    QC.jl

Quality Control (QC) module.
"""

using Statistics
using Distributions

"""
    filter_maf(G, threshold)

Filters SNPs by Minor Allele Frequency.
"""
function filter_maf(G::Matrix{Float64}, threshold::Float64)
    # G is n x m (0, 1, 2)
    freqs = vec(mean(G, dims=1)) ./ 2.0
    maf = min.(freqs, 1.0 .- freqs)
    keep = maf .>= threshold
    return (G[:, keep], keep)
end

"""
    filter_missing(G, threshold)

Filters SNPs/Inds by missing rate.
"""
function filter_missing(G::Matrix{Union{Float64, Missing}}, threshold::Float64; dims=1)
    # dims=1: Filter rows (Inds)
    # dims=2: Filter cols (SNPs)
    
    miss_rate = vec(mean(ismissing.(G), dims=dims))
    keep = miss_rate .<= threshold
    
    if dims == 1
        return (G[keep, :], keep)
    else
        return (G[:, keep], keep)
    end
end

"""
    hwe_test(G)

Hardy-Weinberg Equilibrium test (Chi-square).
"""
function hwe_test(G::Matrix{Float64})
    # G must be 0, 1, 2
    n, m = size(G)
    p_values = zeros(m)
    
    for j in 1:m
        obs_0 = count(x -> x < 0.5, G[:, j])
        obs_1 = count(x -> 0.5 <= x < 1.5, G[:, j])
        obs_2 = count(x -> x >= 1.5, G[:, j])
        
        n_total = obs_0 + obs_1 + obs_2
        p = (2 * obs_2 + obs_1) / (2 * n_total)
        q = 1.0 - p
        
        exp_0 = n_total * q^2
        exp_1 = n_total * 2 * p * q
        exp_2 = n_total * p^2
        
        chisq = (obs_0 - exp_0)^2/exp_0 + (obs_1 - exp_1)^2/exp_1 + (obs_2 - exp_2)^2/exp_2
        
        p_values[j] = 1.0 - cdf(Chisq(1), chisq)
    end
    
    return p_values
end
