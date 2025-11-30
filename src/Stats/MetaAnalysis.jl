"""
    MetaAnalysis.jl

Meta-Analysis tools for GWAS.
"""

using Statistics
using Distributions

"""
    meta_analysis_ivw(betas, ses)

Inverse Variance Weighted (IVW) Meta-analysis.
betas: Vector of effect sizes from k studies.
ses: Vector of standard errors from k studies.
"""
function meta_analysis_ivw(betas::Vector{Float64}, ses::Vector{Float64})
    # Weights w_i = 1 / se_i^2
    weights = 1.0 ./ (ses.^2)
    
    # Weighted mean beta
    # beta_meta = sum(w_i * beta_i) / sum(w_i)
    beta_meta = sum(weights .* betas) / sum(weights)
    
    # Standard error of meta beta
    # se_meta = sqrt(1 / sum(w_i))
    se_meta = sqrt(1.0 / sum(weights))
    
    # Z-score and P-value
    z = beta_meta / se_meta
    p_value = 2.0 * (1.0 - cdf(Normal(), abs(z)))
    
    return (beta=beta_meta, se=se_meta, p_value=p_value)
end
