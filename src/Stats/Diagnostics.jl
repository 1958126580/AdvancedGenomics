"""
    Diagnostics.jl

MCMC Convergence Diagnostics.
Includes Geweke, Effective Sample Size (ESS), and Autocorrelation.
"""

using Statistics
using Distributions

"""
    autocorrelation(chain, max_lag)

Computes autocorrelation for a vector up to max_lag.
"""
function autocorrelation(chain::Vector{Float64}, max_lag::Int)
    n = length(chain)
    mu = mean(chain)
    var_chain = var(chain; corrected=false) # Sample variance
    
    ac = zeros(max_lag + 1)
    
    # Lag 0 is 1.0
    ac[1] = 1.0
    
    for k in 1:max_lag
        # Covariance at lag k
        cov_k = 0.0
        for i in 1:(n - k)
            cov_k += (chain[i] - mu) * (chain[i+k] - mu)
        end
        cov_k /= n
        ac[k+1] = cov_k / var_chain
    end
    
    return ac
end

"""
    effective_sample_size(chain)

Estimates Effective Sample Size (ESS).
ESS = N / (1 + 2 * sum(rho_k))
Summation is truncated when rho_k becomes negative or small.
"""
function effective_sample_size(chain::Vector{Float64})
    n = length(chain)
    max_lag = min(n - 1, 1000) # Limit lag
    rho = autocorrelation(chain, max_lag)
    
    # Sum autocorrelations until sum decreases or rho becomes negative
    sum_rho = 0.0
    for k in 2:length(rho) # rho[1] is lag 0 (1.0)
        if rho[k] < 0.05 # Threshold for noise
            break
        end
        sum_rho += rho[k]
    end
    
    ess = n / (1.0 + 2.0 * sum_rho)
    return ess
end

"""
    geweke_diagnostic(chain; frac1=0.1, frac2=0.5)

Geweke Diagnostic.
Compares mean of first frac1 (10%) and last frac2 (50%) of the chain.
Returns Z-score.
Z = (mean1 - mean2) / sqrt(var1 + var2)
Variances are corrected for autocorrelation (Spectral density at freq 0).
For simplicity here, we use standard error assuming independence if ESS is high,
or simple batch means / overlapping batch means.
Here we implement a simplified version using naive variance, 
but for "Top Level" we should ideally use spectral variance.
Let's use the variance of the means (Standard Error).
SE = SD / sqrt(ESS)
"""
function geweke_diagnostic(chain::Vector{Float64}; frac1=0.1, frac2=0.5)
    n = length(chain)
    n1 = Int(floor(frac1 * n))
    n2 = Int(floor(frac2 * n))
    
    start_2 = n - n2 + 1
    
    chain1 = chain[1:n1]
    chain2 = chain[start_2:end]
    
    m1 = mean(chain1)
    m2 = mean(chain2)
    
    # Variance of the mean
    # var(mean) = var(chain) / ESS
    # We estimate ESS for each part
    ess1 = effective_sample_size(chain1)
    ess2 = effective_sample_size(chain2)
    
    var1 = var(chain1) / ess1
    var2 = var(chain2) / ess2
    
    z = (m1 - m2) / sqrt(var1 + var2)
    
    p_value = 2.0 * (1.0 - cdf(Normal(), abs(z)))
    
    return (z_score=z, p_value=p_value)
end
