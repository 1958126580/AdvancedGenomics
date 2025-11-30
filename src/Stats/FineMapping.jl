"""
    FineMapping.jl

Post-GWAS Fine-mapping tools.
"""

using Statistics

"""
    simple_fine_mapping(z_scores, ld_matrix; prior_var=0.1)

Simple Bayesian Fine-mapping assuming single causal variant (ABF).
Calculates Posterior Inclusion Probability (PIP) for each SNP in the locus.
"""
function simple_fine_mapping(z_scores::Vector{Float64}, ld_matrix::Matrix{Float64}; prior_var::Float64=0.2^2)
    # Approximate Bayes Factor (ABF) for each SNP
    # Wakefield (2009) approximation
    # ABF = sqrt(V / (V + W)) * exp( (z^2/2) * (W / (V + W)) )
    # V = variance of beta_hat approx 1/N? Or derived from Z?
    # Z = beta / se. V = se^2.
    # We don't have V directly, but if we assume standardized effect sizes...
    
    # Simplified: ABF depends on Z score.
    # Let's assume V is small relative to W (prior variance).
    
    # Using Z-score only formulation (assuming N is large)
    # log(ABF) approx Z^2 / 2 + const?
    
    # Let's use a standard simplified ABF formula:
    # ABF_j = exp(Z_j^2 / 2) (very rough, ignores prior width penalty)
    
    # Better:
    # ABF = 1 / sqrt(1 + N*W) * exp( (Z^2/2) * (N*W / (1 + N*W)) )
    # Assume N*W is constant K.
    # ABF propto exp(Z^2 * C)
    
    # For fine-mapping within a locus, we calculate posterior prob:
    # PIP_j = ABF_j / sum(ABF_k)
    
    # Let's use the Z-score directly.
    # log_ABF = z_scores.^2 ./ 2.0
    
    # To avoid overflow, work in log space
    log_abf = z_scores.^2 ./ 2.0
    max_log = maximum(log_abf)
    
    # exp(log_abf - max_log)
    abf_scaled = exp.(log_abf .- max_log)
    
    pip = abf_scaled ./ sum(abf_scaled)
    
    return pip
end
