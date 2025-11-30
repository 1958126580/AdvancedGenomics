"""
    Causal.jl

Causal Inference module.
Implements Mendelian Randomization (MR) methods.
"""

using Statistics
using LinearAlgebra
using Distributions

"""
    mr_ivw(beta_exp, se_exp, beta_out, se_out)

Inverse Variance Weighted (IVW) method for Mendelian Randomization.
beta_exp: Effect of IVs on exposure
beta_out: Effect of IVs on outcome
"""
function mr_ivw(beta_exp::Vector{Float64}, se_exp::Vector{Float64}, beta_out::Vector{Float64}, se_out::Vector{Float64})
    # Weights = 1 / se_out^2 (Simple IVW)
    # Or Weights = 1 / (se_out^2 / beta_exp^2 + ...)
    
    # Standard IVW: Weighted regression of beta_out on beta_exp through origin
    weights = 1.0 ./ se_out.^2
    
    # Weighted Least Squares: beta_causal = sum(w * b_out * b_exp) / sum(w * b_exp^2)
    numerator = sum(weights .* beta_out .* beta_exp)
    denominator = sum(weights .* beta_exp.^2)
    
    beta_causal = numerator / denominator
    
    # Standard Error
    # se_causal = sqrt(1 / sum(w * b_exp^2))
    se_causal = sqrt(1.0 / denominator)
    
    p_value = 2.0 * (1.0 - cdf(Normal(), abs(beta_causal / se_causal)))
    
    return (beta=beta_causal, se=se_causal, p_value=p_value)
end

"""
    mr_egger(beta_exp, se_exp, beta_out, se_out)

MR-Egger regression.
Allows for pleiotropy (intercept != 0).
"""
function mr_egger(beta_exp::Vector{Float64}, se_exp::Vector{Float64}, beta_out::Vector{Float64}, se_out::Vector{Float64})
    weights = 1.0 ./ se_out.^2
    
    # Weighted regression of beta_out on beta_exp with intercept
    # Y = beta_out, X = [1 beta_exp]
    n = length(beta_out)
    X = hcat(ones(n), beta_exp)
    W = Diagonal(weights)
    
    # (X'WX)^-1 X'WY
    XtWX = X' * W * X
    XtWY = X' * W * beta_out
    
    theta = XtWX \ XtWY
    
    intercept = theta[1]
    beta_causal = theta[2]
    
    # SE
    resid = beta_out - X * theta
    sigma_sq = sum(weights .* resid.^2) / (n - 2)
    var_theta = inv(XtWX) * sigma_sq
    
    se_intercept = sqrt(var_theta[1, 1])
    se_beta = sqrt(var_theta[2, 2])
    
    p_intercept = 2.0 * (1.0 - cdf(TDist(n-2), abs(intercept / se_intercept)))
    p_beta = 2.0 * (1.0 - cdf(TDist(n-2), abs(beta_causal / se_beta)))
    
    return (
        beta=beta_causal, se=se_beta, p_value=p_beta,
        intercept=intercept, se_intercept=se_intercept, p_intercept=p_intercept
    )
end
