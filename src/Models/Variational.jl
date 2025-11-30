"""
    Variational.jl

Variational Inference (VI) for Genomic Prediction.
Implements Mean-Field Variational Bayes for BayesC.
"""

using LinearAlgebra
using Statistics
using Distributions

"""
    run_vi_bayesc(y, X; max_iter=100, tol=1e-5)

Runs Variational Inference for BayesC.
Approximates posterior P(beta, delta | y) with Q(beta)Q(delta).
"""
function run_vi_bayesc(y::Vector{Float64}, X::Matrix{Float64}; max_iter::Int=100, tol::Float64=1e-5)
    n, p = size(X)
    
    # Variational Parameters
    # For each marker j:
    # alpha[j]: probability delta_j = 1
    # mu[j]: mean of beta_j | delta_j = 1
    # s2[j]: variance of beta_j | delta_j = 1
    
    alpha = fill(0.5, p)
    mu = zeros(p)
    s2 = fill(0.01, p)
    
    # Hyperparameters (assumed fixed for simplicity or updated via VI)
    sigma_e2 = var(y) / 2.0
    sigma_g2 = 0.01
    pi_prior = 0.05
    
    XtX_diag = vec(sum(X.^2, dims=1))
    XtY = X' * y
    
    # Residual (Expected)
    # E[y_pred] = X * (alpha .* mu)
    y_pred = X * (alpha .* mu)
    
    for iter in 1:max_iter
        max_diff = 0.0
        
        for j in 1:p
            # Remove effect of j
            # y_corr = y - y_pred + X_j * E[beta_j]
            # E[beta_j] = alpha_j * mu_j
            
            # Efficient update without full residual calculation?
            # We need X_j' * (y - sum_{k!=j} X_k E[beta_k])
            # = X_j' * y - X_j' * X * E[beta] + X_j' * X_j * E[beta_j]
            
            # This is O(np) inside loop -> O(np^2) total. Too slow.
            # Use residual vector update.
            
            old_exp_beta = alpha[j] * mu[j]
            
            # Update residual
            # r_min_j = y - y_pred + X[:, j] * old_exp_beta
            # rhs = dot(X[:, j], r_min_j)
            
            # Faster: rhs = XtY[j] - (XtX[j, :] * E[beta]) + XtX[j,j]*old_exp_beta
            # Still O(p).
            # Let's use the residual vector `e = y - y_pred`
            
            e = y - y_pred # This should be updated incrementally
            rhs = dot(X[:, j], e) + XtX_diag[j] * old_exp_beta
            
            # Update Q(beta_j | delta_j=1) ~ N(mu_j, s2_j)
            # s2_j = (XtX_j / sigma_e2 + 1/sigma_g2)^-1
            s2[j] = 1.0 / (XtX_diag[j] / sigma_e2 + 1.0 / sigma_g2)
            mu[j] = s2[j] * (rhs / sigma_e2)
            
            # Update Q(delta_j) ~ Bernoulli(alpha_j)
            # log(alpha / (1-alpha)) = log(pi/(1-pi)) + 0.5*log(s2_j) - 0.5*log(sigma_g2) + 0.5*mu_j^2/s2_j
            
            log_odds = log(pi_prior / (1.0 - pi_prior)) + 0.5 * log(s2[j] / sigma_g2) + 0.5 * (mu[j]^2 / s2[j])
            new_alpha = 1.0 / (1.0 + exp(-log_odds))
            
            # Convergence check
            diff = abs(new_alpha - alpha[j])
            if diff > max_diff
                max_diff = diff
            end
            
            alpha[j] = new_alpha
            
            # Update y_pred incrementally
            new_exp_beta = alpha[j] * mu[j]
            y_pred += X[:, j] * (new_exp_beta - old_exp_beta)
        end
        
        if max_diff < tol
            break
        end
    end
    
    return (alpha=alpha, mu=mu, beta_hat=alpha .* mu)
end
