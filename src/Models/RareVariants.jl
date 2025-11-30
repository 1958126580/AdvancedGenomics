"""
    RareVariants.jl

Methods for analyzing rare variants.
Implements SKAT (Sequence Kernel Association Test).
"""

using LinearAlgebra
using Statistics
using Distributions

"""
    skat_kernel(G::Matrix, weights::Vector)

Computes the weighted linear kernel for SKAT.
K = G * W * W * G'
"""
function skat_kernel(G::Matrix{Float64}, weights::Vector{Float64})
    W = Diagonal(weights)
    GW = G * W
    return GW * GW'
end

"""
    run_skat(y::Vector, G::Matrix, X::Matrix; weights=nothing)

Runs the Sequence Kernel Association Test (SKAT).
y: phenotype (continuous)
G: genotype matrix (n x m) for the region/gene
X: covariates (n x p)
"""
function run_skat(y::Vector{Float64}, G::Matrix{Float64}, X::Matrix{Float64}; weights=nothing)
    n, m = size(G)
    
    # Default weights (Beta(1, 25) density evaluated at MAF)
    if isnothing(weights)
        mafs = vec(mean(G, dims=1)) ./ 2.0
        weights = Beta(1, 25).(mafs)
    end
    
    # Null model: y = Xb + e
    # Estimate sigma_e2 under null
    b_null = (X' * X) \ (X' * y)
    res = y - X * b_null
    sigma_e2 = var(res)
    
    # Projection matrix P0 = V^-1 - V^-1 X (X' V^-1 X)^-1 X' V^-1
    # For linear model V = I * sigma_e2
    # P0 = (I - H) / sigma_e2
    H = X * inv(X' * X) * X'
    P0 = (I(n) - H) ./ sigma_e2
    
    # Q statistic
    K = skat_kernel(G, weights)
    Q = dot(res, K, res) # Quadratic form y' P K P y ? No, just res' K res?
    # Wu et al. (2011): Q = (y - Xb)' K (y - Xb)
    
    # P-value calculation using Davies method (approximation via eigenvalues)
    # Q ~ sum(lambda_i * chi2_1)
    # lambda_i are eigenvalues of P0^{1/2} K P0^{1/2}
    
    PKP = P0 * K
    lambdas = eigvals(PKP)
    lambdas = real.(lambdas[lambdas .> 1e-8]) # Keep positive eigenvalues
    
    # Simple approximation using Satterthwaite if Davies not available
    # mean_Q = sum(lambdas)
    # var_Q = 2 * sum(lambdas.^2)
    # df = 2 * mean_Q^2 / var_Q
    # p_val = 1 - cdf(Chisq(df), Q * 2 * mean_Q / var_Q) -- this is rough
    
    # Using a simpler approximation for now:
    # In a real package, we'd wrap the C implementation of Davies algorithm.
    # Here we use a normal approximation for very large degrees of freedom or simple ChiSq
    
    mean_Q = sum(lambdas)
    var_Q = 2 * sum(lambdas.^2)
    df_eff = 2 * mean_Q^2 / var_Q
    
    # Scale Q to match ChiSq distribution
    Q_scaled = (Q - mean_Q) / sqrt(var_Q) * sqrt(2 * df_eff) + df_eff
    
    p_value = 1.0 - cdf(Chisq(df_eff), Q_scaled)
    
    return (Q=Q, p_value=p_value, df=df_eff)
end

"""
    run_burden_test(y, G, X)

Burden Test (Collapsing method) for rare variants.
Collapses rare variants into a single score (burden) per individual.
"""
function run_burden_test(y::Vector{Float64}, G::Matrix{Float64}, X::Matrix{Float64}; threshold=0.05)
    n, m = size(G)
    
    # 1. Collapse variants
    # Simple burden: Sum of minor alleles
    # Weighted burden: Sum of w_j * G_ij
    # Here we use simple sum (CAST-like) or frequency weighted
    
    burden = vec(sum(G, dims=2))
    
    # 2. Regression
    # y = Xb + beta_burden * burden + e
    X_new = hcat(X, burden)
    
    # Solve OLS
    beta = X_new \ y
    resid = y - X_new * beta
    sigma2 = var(resid)
    
    # Standard Error of beta_burden
    XtX_inv = inv(X_new' * X_new)
    se_burden = sqrt(sigma2 * XtX_inv[end, end])
    
    t_stat = beta[end] / se_burden
    p_value = 2.0 * (1.0 - cdf(TDist(n - size(X_new, 2)), abs(t_stat)))
    
    return (beta=beta[end], se=se_burden, p_value=p_value)
end
